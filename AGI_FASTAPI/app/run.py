import logging
import uuid

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends, APIRouter

from fastapi.responses import JSONResponse
import numpy as np
import re
import textwrap
import markdown
from typing import Optional, List, Dict, Union, Type

from fastapi_cache import FastAPICache
from pydantic import BaseModel,Field
from cachetools import cached, TTLCache
import requests
from sqlalchemy.orm import Session
from models import EnvironmentDB, AgentDB  # Import your models
from database import get_db, DatabaseService  # Import your database utilities
import pandas as pd
import shutil
from openai import OpenAI
import os
import tempfile
import io
import sys
from sqlalchemy import create_engine
from plotly.graph_objs import Figure
from typing import Dict, Any

router = APIRouter(prefix="/openai", tags=["openai_run"])
logger = logging.getLogger(__name__)
# Cache setup
cache = TTLCache(maxsize=1000, ttl=3600)


# Updated Pydantic models for request validation
class AgentRequest(BaseModel):
    agent_id: int = Field(..., description="ID of the agent to use")
    prompt: Optional[str] = Field(
        None,
        description="Optional text prompt for the agent",
        example="Explain this document"
    )
    url: Optional[str] = Field(
        None,
        description="Optional URL for agents that work with web content",
        example="https://example.com"
    )

class FileUploadRequest(AgentRequest):
    file: Optional[UploadFile] = Field(
        None,
        description="Single file upload (PDF, DOC, DOCX, TXT, JPG, PNG, JPEG)",
        example="document.pdf"
    )
    class Config:
        # This is important for file uploads in Pydantic models
        arbitrary_types_allowed = True


# Helper functions
def extract_travel_details(prompt: str) -> Union[Dict[str, str], tuple]:
    """Extracts destination and number of days from a user's travel request prompt."""
    try:
        prompt = prompt.strip()
        days_match = re.search(r'\bfor (\d+) (?:days|nights)\b', prompt, re.IGNORECASE)
        destination_match = re.search(r'\bto ([A-Za-z ,]+)\b', prompt, re.IGNORECASE)

        days = int(days_match.group(1)) if days_match else None
        destination = destination_match.group(1).strip() if destination_match else None

        if destination:
            destination = destination.split(',')[0].strip()

        if not destination or not days:
            return {"error": "Could not extract destination or number of days from prompt."}

        return destination, days
    except Exception as e:
        return {"error": f"Error processing prompt: {str(e)}"}


def differentiate_url(url: str) -> str:
    """Differentiates between YouTube and regular website URLs."""
    youtube_patterns = [
        r"^(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+(&\S*)?$",
        r"^(https?://)?(www\.)?youtu\.be/[\w-]+$",
        r"^(https?://)?(www\.)?youtube\.com/playlist\?list=[\w-]+(&\S*)?$",
        r"^(https?://)?(www\.)?youtube\.com/channel/[\w-]+$",
        r"^(https?://)?(www\.)?youtube\.com/c/[\w-]+$",
        r"^(https?://)?(www\.)?youtube\.com/user/[\w-]+$",
    ]

    for pattern in youtube_patterns:
        if re.match(pattern, url, re.IGNORECASE):
            return "YouTube"
    return "Website"


def extract_topic_or_query(user_prompt: str) -> str:
    """Determines if the user input is a topic name or a query."""
    topic_pattern = r"^[A-Za-z\s]+$"
    question_words = ["what", "why", "how", "who", "when", "where", "is", "can", "does", "explain", "?"]

    user_prompt_lower = user_prompt.lower().strip()

    if any(word in user_prompt_lower for word in question_words) or not re.match(topic_pattern, user_prompt):
        return "query"
    return "topic"


def markdown_to_html(markdown_text: str) -> str:
    """Converts markdown text to HTML."""
    return markdown.markdown(markdown_text, extensions=["extra"])


# Main endpoint with database integration
@router.post("/run_openai_environment/")
async def run_openai_environment(
        request_data: FileUploadRequest = Depends(),
        db: Session = Depends(get_db)
):
    # Access fields via request_data:
    agent_id = request_data.agent_id
    prompt = request_data.prompt
    url = request_data.url
    file = request_data.file
    print("DEBUG: Entering run_openai_environment endpoint")
    try:
        print(f"DEBUG: Received parameters - agent_id: {agent_id}, prompt: {prompt}, url: {url}, file: {file}")

        if file:
            print("DEBUG: File upload detected")

            # Updated allowed content types
            allowed_content_types = [
                'application/pdf',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'text/plain',
                'image/jpeg',
                'image/png',
                'image/jpg',
                'text/csv',
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            ]

            print(f"DEBUG: File content type: {file.content_type}")
            print(f"DEBUG: File filename: {file.filename}")

            if file.content_type not in allowed_content_types:
                _, file_ext = os.path.splitext(file.filename)
                file_ext = file_ext.lower().lstrip('.')  # e.g., '.xlsx' → 'xlsx'
                print(f"DEBUG: File extension: {file_ext}")

                allowed_extensions = ['pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'csv', 'xls', 'xlsx']

                if file_ext not in allowed_extensions:
                    print("DEBUG: Unsupported file type")
                    raise HTTPException(
                        status_code=400,
                        detail="Unsupported file type. Please upload PDF, DOC, DOCX, TXT, JPG, JPEG, PNG, CSV, XLS, or XLSX files."
                    )
                else:
                    print("DEBUG: File type allowed by extension")
            else:
                print("DEBUG: File type allowed by content type")
        # Initialize database service
        print("DEBUG: Initializing database service")
        db_service = DatabaseService(db)

        # Retrieve agent details
        print(f"DEBUG: Retrieving agent with ID: {agent_id}")
        agent = db_service.get_agent(int(agent_id))
        if not agent:
            print("DEBUG: Agent not found")
            raise HTTPException(status_code=404, detail="Agent not found")
        print(f"DEBUG: Found agent - backend_id: {agent.backend_id}")

        # Retrieve environment details
        print(f"DEBUG: Retrieving environment for agent with env_id: {agent.env_id}")
        env_details = db_service.get_environment(agent.env_id)
        if not env_details:
            print("DEBUG: Environment not found")
            raise HTTPException(status_code=404, detail="Environment not found")
        print("DEBUG: Environment details retrieved")

        openai_api_key = env_details.api_key
        print("DEBUG: Initializing OpenAI client")
        client = OpenAI(api_key=openai_api_key)

        result = None
        BLOG_TOOL_IDS = ['blog_post', 'audio_blog', 'video_blog', 'youtube_blog']

        # 1. Text_to_sql application
        if file and prompt and 'text_to_sql' in agent.backend_id:
            print("DEBUG: Processing text_to_sql request")
            result = await gen_response(file, prompt, client)
            print(f"DEBUG: gen_response result: {result}")

            if not result:
                print("DEBUG: gen_response returned None")
                raise HTTPException(status_code=500, detail="gen_response() returned None")

            if "chartData" in result:
                print("DEBUG: Returning chart data")
                return JSONResponse(content={"chartData": result["chartData"]})
            elif "answer" in result:
                print("DEBUG: Returning answer")
                return JSONResponse(content={"answer": markdown_to_html(result["answer"])})
            else:
                print("DEBUG: No valid output from gen_response")
                raise HTTPException(status_code=400, detail="No valid output from gen_response")

        # 2. Chat to doc within specific page numbers and querying
        elif file and prompt and 'chat_to_doc_within_page_range' in agent.backend_id:
            print("DEBUG: Processing chat_to_doc_within_page_range request")
            try:
                if not file.filename:
                    print("DEBUG: No file uploaded")
                    raise HTTPException(status_code=400, detail="No file uploaded")
                if not prompt.strip():
                    print("DEBUG: Empty prompt")
                    raise HTTPException(status_code=400, detail="Prompt cannot be empty")

                print("DEBUG: Processing document question answering")
                result = await document_question_answering(
                    api_key=openai_api_key,
                    uploaded_file=file,
                    query=prompt.strip()
                )
                print(f"DEBUG: document_question_answering result: {result}")

                if "answer" in result:
                    print("DEBUG: Returning answer")
                    return JSONResponse(content={
                        "answer": markdown_to_html(result["answer"])
                    })
                elif "error" in result:
                    print(f"DEBUG: Error in result: {result['error']}")
                    raise HTTPException(status_code=400, detail=result["error"])
                else:
                    print("DEBUG: Document processing failed")
                    raise HTTPException(
                        status_code=500,
                        detail="Document processing failed to return a valid answer"
                    )

            except HTTPException as he:
                print(f"DEBUG: HTTPException occurred: {he}")
                raise he
            except Exception as e:
                print(f"DEBUG: Exception in document processing: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Document processing error: {str(e)}"
                )

        # 3. Travel planner agent
        elif prompt and 'travel_planner' in agent.backend_id:
            print("DEBUG: Processing travel_planner request")
            # travel_planner_agent = TravelPlannerAgent(openai_api_key)
            # destination, days = travel_planner_agent.parse_user_input(prompt)
            # print(f"DEBUG: Parsed travel details - destination: {destination}, days: {days}")
            result = await generate_travel_plan(prompt, openai_api_key)
            print(f"DEBUG: generate_travel_plan result: {result}")
            if not result:
                print("DEBUG: generate_travel_plan returned None")
                raise HTTPException(status_code=500, detail="generate_travel_plan() returned None")
            if "answer" in result:
                print("DEBUG: Returning travel plan")
                return JSONResponse(content={"answer": markdown_to_html(result["answer"])})

        # 4. Medical Diagnosis Agent
        elif file and 'medical_diagnosis' in agent.backend_id:
            print("DEBUG: Processing medical_diagnosis request")
            file_content = await file.read()
            print("DEBUG: File content read successfully")
            result = run_medical_diagnosis(openai_api_key, file_content.decode("utf-8"))
            print(f"DEBUG: run_medical_diagnosis result: {result}")

            if not result:
                print("DEBUG: run_medical_diagnosis returned None")
                raise HTTPException(status_code=500, detail="run_medical_diagnosis() returned None")

            if "report" in result:
                print("DEBUG: Returning medical report")
                return JSONResponse(content={"answer": markdown_to_html(result["report"])})

        # 5. EduGPT
        elif prompt and "edu_gpt" in agent.backend_id:
            print("DEBUG: Processing edu_gpt request")
            prompt_type = extract_topic_or_query(prompt)
            print(f"DEBUG: Extracted prompt type: {prompt_type}")
            try:
                if prompt_type == "topic":
                    print("DEBUG: Starting new learning session")
                    result = await start_learning(request, prompt, openai_api_key)
                    print("DEBUG: Learning session started")

                    return JSONResponse(content={
                        "answer": markdown_to_html(result["syllabus"]),
                        "cache_key": result["cache_key"]
                    })

                elif prompt_type == "query":
                    print("DEBUG: Continuing existing conversation")
                    cache_key = request.headers.get("X-Cache-Key") or request.query_params.get("cache_key")
                    print(f"DEBUG: Using cache key: {cache_key}")

                    if not cache_key:
                        print("DEBUG: Cache key missing")
                        raise HTTPException(
                            status_code=400,
                            detail="Cache key required. Please start learning first by providing a topic."
                        )

                    result = await chat_with_agent(cache_key, prompt, openai_api_key)
                    print("DEBUG: Chat with agent completed")

                    return JSONResponse(content={
                        "answer": markdown_to_html(result["assistant_response"]),
                        "cache_key": result["cache_key"]
                    })

                else:
                    print("DEBUG: Invalid prompt type")
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid prompt type. Please provide either a topic to learn or a question."
                    )

            except HTTPException as he:
                print(f"DEBUG: HTTPException in edu_gpt: {he}")
                raise he
            except Exception as e:
                print(f"DEBUG: Exception in edu_gpt: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing education request: {str(e)}"
                )

        # 6. Medical Image processing
        elif file and 'image_processing' in agent.backend_id:
            print("DEBUG: Processing medical image")
            result = medical_image_analysis(openai_api_key, file)
            print(f"DEBUG: medical_image_analysis result: {result}")
            if not result:
                print("DEBUG: medical_image_analysis returned None")
                raise HTTPException(status_code=500, detail="medical_image_analysis() returned None")

            if "result" in result:
                print("DEBUG: Returning image analysis result")
                return JSONResponse(content={"answer": markdown_to_html(result["result"])})

        # 7. Image answering
        elif file and prompt and 'image_answering' in agent.backend_id:
            print("DEBUG: Processing image answering")
            result = visual_question_answering(openai_api_key, file, prompt)
            print(f"DEBUG: visual_question_answering result: {result}")
            if not result:
                print("DEBUG: visual_question_answering returned None")
                raise HTTPException(status_code=500, detail="visual_question_answering() returned None")

            if "answer" in result:
                print("DEBUG: Returning visual answer")
                return JSONResponse(content={"answer": markdown_to_html(result["answer"])})

        # 8. Blog agents with URL
        elif url and prompt:
            print("DEBUG: Processing blog agent with URL")
            url_type = differentiate_url(url)
            print(f"DEBUG: URL type: {url_type}")

            if url_type == "YouTube":
                if any(tool_id in agent.backend_id for tool_id in BLOG_TOOL_IDS):
                    print("DEBUG: Generating blog from YouTube URL")
                    result = generate_blog_from_yt_url(prompt, url, 'blog_post', openai_api_key)
                elif 'linkedin_post' in agent.backend_id:
                    print("DEBUG: Generating LinkedIn post from YouTube URL")
                    result = generate_blog_from_yt_url(prompt, url, 'linkedin_post', openai_api_key)
            else:
                if any(tool_id in agent.backend_id for tool_id in BLOG_TOOL_IDS):
                    print("DEBUG: Generating blog from URL")
                    result = generate_blog_from_url(prompt, url, 'blog_post', openai_api_key)
                elif 'linkedin_post' in agent.backend_id:
                    print("DEBUG: Generating LinkedIn post from URL")
                    result = generate_blog_from_url(prompt, url, 'linkedin_post', openai_api_key)

            if isinstance(result, dict):
                print("DEBUG: Returning blog content")
                response_data = markdown_to_html(result["content"])
                return JSONResponse(content={"answer": response_data})

        # 9. Blog Generation through file
        elif file and prompt:
            print("DEBUG: Processing blog generation from file")
            if any(tool_id in agent.backend_id for tool_id in BLOG_TOOL_IDS):
                print("DEBUG: Generating blog from file")
                result = generate_blog_from_file(prompt, file, 'blog_post', openai_api_key)
            elif 'linkedin_post' in agent.backend_id:
                print("DEBUG: Generating LinkedIn post from file")
                result = generate_blog_from_file(prompt, file, 'linkedin_post', openai_api_key)

            if isinstance(result, dict):
                print("DEBUG: Returning blog content from file")
                response_data = markdown_to_html(result["content"])
                return JSONResponse(content={"answer": response_data})

        # 10. MCQ Generator
        elif prompt and 'mcq_generator' in agent.backend_id:
            print("DEBUG: Processing MCQ generation")
            result = mcq_generation(openai_api_key, prompt)
            print(f"DEBUG: mcq_generation result: {result}")
            if "answer" in result:
                print("DEBUG: Returning MCQ result")
                return JSONResponse(content={"answer": markdown_to_html(result["answer"])})

        # 11. Synthetic_data_generator
        elif 'synthetic_data_generation' in agent.backend_id:
            print("DEBUG: Processing synthetic data generation")
            if prompt and file:
                print("DEBUG: Handling synthetic data from Excel with prompt")
                result = handle_synthetic_data_from_excel(file, openai_api_key, prompt)
            elif file:
                print("DEBUG: Handling fill missing data")
                result = handle_fill_missing_data(file, openai_api_key)
            else:
                print("DEBUG: Handling synthetic data for new data")
                result = handle_synthetic_data_for_new_data(prompt, openai_api_key)

            print(f"DEBUG: Synthetic data result: {result}")
            if result and "data" in result:
                print("DEBUG: Returning synthetic data")
                return JSONResponse(content=result)
            elif result and "error" in result:
                print(f"DEBUG: Synthetic data error: {result['error']}")
                raise HTTPException(status_code=400, detail=result["error"])
            else:
                print("DEBUG: Unexpected error in synthetic data")
                raise HTTPException(status_code=500, detail="An unexpected error occurred.")

        # Final else condition
        else:
            print("DEBUG: No matching agent backend_id found")
            raise HTTPException(status_code=500, detail="Unexpected response format from the analyzer")

    except ValueError as e:
        print(f"DEBUG: ValueError occurred: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid agent ID format: {str(e)}")
    except Exception as e:
        print(f"DEBUG: Unexpected exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 1.Text-to-sql
# Database configuration
USER = 'test_owner'
PASSWORD = 'tcWI7unQ6REA'
HOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech:5432'
DATABASE = 'test'

async def gen_response(upload_file, query: str, client) -> Dict[str, Any]:
    print("Getting into the function")
    # Save the uploaded file to a temporary path
    suffix = os.path.splitext(upload_file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await upload_file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    table_name = os.path.splitext(os.path.basename(upload_file.filename))[0]
    print(table_name)

    # Convert file to SQL & DataFrame
    df = file_to_sql(tmp_file_path, table_name, USER, PASSWORD, HOST, DATABASE)
    print(df.head(5))

    # Delete temp file
    os.unlink(tmp_file_path)

    csv_metadata = {"columns": df.columns.tolist()}
    metadata_str = ", ".join(csv_metadata["columns"])
    if not query:
        return {"error": "No query provided"}

    # Check for graph-related query
    graph_keywords = [
        "plot", "graph", "visualize", "visualization", "scatter", "bar chart",
        "line chart", "histogram", "pie chart", "bubble chart", "heatmap", "box plot",
        "generate chart", "create graph", "draw", "trend", "correlation"
    ]
    print("if condition............")
    if any(k in query.lower() for k in graph_keywords):
        prompt_eng = (
            f"You are an AI specialized in data analytics and visualization."
            f"Data used for analysis is stored in a pandas DataFrame named `df`. "
            f"The DataFrame `df` contains the following attributes: {metadata_str}. "
            f"Based on the user's query, generate Python code using Plotly to create the requested type of graph "
            f"(e.g., bar, pie, scatter, etc.) using the data in the DataFrame `df`. "
            f"The graph must utilize the data within `df` as appropriate for the query. "
            f"If the user does not specify a graph type, decide whether to generate a line or bar graph based on the situation."
            f"Every graph must include a title, axis labels (if applicable), and appropriate colors for better visualization."
            f"The graph must have a white background for both the plot and paper. "
            f"The code must output a Plotly 'Figure' object stored in a variable named 'fig', "
            f"and include 'data' and 'layout' dictionaries compatible with React. "
            f"The user asks: {query}."
        )
        chat = generate_code(prompt_eng, client)

        if 'import' in chat:
            namespace = {'df': df}
            try:
                exec(chat, namespace)
                fig = namespace.get("fig")

                if fig and isinstance(fig, Figure):
                    def make_serializable(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: make_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [make_serializable(v) for v in obj]
                        return obj

                    chart_data = make_serializable(fig.to_plotly_json())
                    return {"chartData": chart_data}
                else:
                    return {"message": "No valid Plotly figure found."}
            except Exception as e:
                return {"message": f"Error executing code: {str(e)}"}
        else:
            return {"message": "AI response does not contain valid code."}
    else:
        print("else condition..........")
        prompt_eng = (
            f"""
                        You are a Python expert focused on answering user queries about data preprocessing and analysis. Always strictly adhere to the following rules:

                        1. Data-Driven Queries:
                            If the user's query is related to data processing or analysis, assume the `df` DataFrame in memory contains the actual uploaded data  with the following columns: {metadata_str}.

                            For such queries:
                            - Generate Python code that directly interacts with the `df` DataFrame to provide accurate results strictly based on the data in the dataset.
                            - Do not make any assumptions or provide any example outputs.
                            - Ensure all answers are derived from actual calculations on the `df` DataFrame.
                            - Include concise comments explaining key steps in the code.
                            - Exclude any visualization, plotting, or assumptions about the data.

                            Example:

                            Query: "How many rows have 'Column1' > 100?"
                            Response:
                            ```python
                            # Count rows where 'Column1' > 100
                            count_rows = df[df['Column1'] > 100].shape[0]

                            # Output the result
                            print(count_rows)
                            ```

                        2. Invalid or Non-Data Queries:
                            If the user's query is unrelated to data processing or analysis, or it cannot be answered using the dataset, respond with an appropriate print statement indicating the limitation. For example:

                            Query: "What is AI?"
                            Response:
                            ```python
                            print("This question is unrelated to the uploaded data. Please ask a data-specific query.")
                            ```

                        3. Theoretical Concepts:
                            If the user asks about theoretical concepts in data science or preprocessing (e.g., normalization, standardization), respond with a concise explanation. Keep the response focused and accurate.

                            Example:

                            Query: "What is normalization in data preprocessing?"
                            Response:
                            ```python
                            print("Normalization is a data preprocessing technique used to scale numeric data within a specific range, typically [0, 1], to ensure all features contribute equally to the model.")
                            ```

                        User query: {query}.
                    """
        )
        code = generate_code(prompt_eng, client)
        result = execute_py_code(code, df)
        return {"answer": result}



def file_to_sql(file_path: str, table_name: str, user: str, password: str, host: str, db_name: str) -> pd.DataFrame:
    try:
        engine = create_engine(f"postgresql://{user}:{password}@{host}/{db_name}")

        file_extension = os.path.splitext(file_path)[-1].lower()
        if file_extension == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide an Excel (.xlsx) or CSV (.csv) file.")

        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


def generate_code(prompt_eng: str, client: OpenAI) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_eng}
            ]
        )

        all_text = response.choices[0].message.content
        if "```python" in all_text:
            code_start = all_text.find("```python") + 9
            code_end = all_text.find("```", code_start)
            code = all_text[code_start:code_end]
        else:
            code = all_text
        return code
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating code: {str(e)}")


def execute_py_code(code: str, df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    sys.stdout = buffer
    local_vars = {'df': df}

    try:
        exec(code, globals(), local_vars)
        output = buffer.getvalue().strip()

        if not output:
            last_line = code.strip().split('\n')[-1]
            if not last_line.startswith(('print', 'return')):
                output = eval(last_line, globals(), local_vars)
                print(output)
    except Exception as e:
        output = f"Error executing code: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__

    return str(output)


# 2.ATS Resume tracker
from wyge.prebuilt_agents.resume_analyser import ResumeAnalyzer


async def analyze_resume(job_description: str, resume_files: Union[UploadFile, List[UploadFile]], api_key: str) -> Dict[
    str, Any]:
    try:
        if not job_description or not resume_files:
            raise HTTPException(status_code=400, detail="Missing job description or resume files")

        # Initialize the Resume Analyzer
        analyzer = ResumeAnalyzer(api_key)

        # Check if it's a single resume or multiple resumes
        if isinstance(resume_files, list):
            # Multiple resume analysis
            results = []
            for resume_file in resume_files:
                # Read file content
                file_content = await resume_file.read()

                # Extract text from the uploaded resume
                resume_text = analyzer.extract_text_from_pdf(io.BytesIO(file_content))

                # Analyze the resume against the job description
                result = analyzer.analyze_resume(resume_text, job_description)

                # Add filename to the result for identification
                result["filename"] = resume_file.filename
                results.append(result)

            # Create a summary table
            summary_data = []
            for i, result in enumerate(results):
                summary_data.append({
                    "Rank": i + 1,
                    "Resume": result.get("filename", "Unknown"),
                    "Match %": result.get("JD Match", "0%"),
                    "Missing Keywords": ", ".join(result.get("MissingKeywords", [])) if result.get(
                        "MissingKeywords") else "None"
                })

            return {
                "summary": summary_data,
                "detailed_results": results
            }
        else:
            # Single resume analysis
            # Read file content
            file_content = await resume_files.read()

            # Extract text from the uploaded resume
            resume_text = analyzer.extract_text_from_pdf(io.BytesIO(file_content))

            result = analyzer.analyze_resume(resume_text, job_description)

            return {
                "JD Match": result.get("JD Match", "0%"),
                "Missing Keywords": result.get("MissingKeywords", []),
                "Profile Summary": result.get("Profile Summary", "No summary available"),
                "Suggestions": result.get("Suggestions", [])
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 3.Chat to doc
from wyge.prebuilt_agents.rag import RAGApplication


async def document_question_answering(api_key: str, uploaded_file: UploadFile, query: str) -> Dict[str, Any]:
    try:
        # Validate inputs
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        if not uploaded_file:
            raise HTTPException(status_code=400, detail="Document file is required")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.filename)[1]) as tmp_file:
            file_content = await uploaded_file.read()
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            # Parse page range (default 0-25)
            page_range = "0-25"
            parsed_page_range = None
            if page_range:
                try:
                    start, end = map(int, page_range.split('-'))
                    parsed_page_range = (start, end)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid page range format. Use 'start-end'"
                    )

            # Initialize RAG application
            rag_app = RAGApplication(
                file_paths=[tmp_file_path],
                openai_api_key=api_key,
                page_range=parsed_page_range
            )
            response = rag_app.query(query.strip())

            # Format response
            if isinstance(response, dict) and "answer" in response:
                return {"answer": response["answer"]}
            elif hasattr(response, "content"):
                return {"answer": response.content}
            else:
                return {"answer": str(response)}

        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


# 4.Travel planner agent
from wyge.prebuilt_agents.travel_planner import TravelPlannerAgent
async def generate_travel_plan(prompt: str, openai_api_key: str) -> dict:

        openai_api_key=openai_api_key
        # Initialize travel planner agent
        agent = TravelPlannerAgent(openai_api_key=openai_api_key)
        print("Agent Initialised")
        # Generate travel plan
        # prompt = f"I need a detailed travel itinerary for {destination.strip()} for {days} days"
        travel_plan = agent.generate_travel_plan(prompt)
        print(travel_plan)
        return {"answer": travel_plan}



# 5.Medical Diagnosis Agent
from wyge.prebuilt_agents.medical_diagnosis import Cardiologist, Psychologist, Pulmonologist


async def run_medical_diagnosis(api_key: str, medical_report: str) -> Dict[str, Any]:
    try:
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        if not medical_report or not medical_report.strip():
            raise HTTPException(status_code=400, detail="Medical report cannot be empty")

        specialists = {
            "Cardiologist": Cardiologist(medical_report, api_key),
            "Psychologist": Psychologist(medical_report, api_key),
            "Pulmonologist": Pulmonologist(medical_report, api_key)
        }

        diagnoses = {}
        report_content = "## Medical Diagnosis Report\n\n"

        for specialist, agent in specialists.items():
            try:
                diagnosis = agent.run()
                diagnoses[specialist] = diagnosis
                report_content += f"### {specialist}\n\n{diagnosis}\n\n"
            except Exception as e:
                diagnoses[specialist] = f"Diagnosis failed: {str(e)}"
                report_content += f"### {specialist}\n\nDiagnosis unavailable: {str(e)}\n\n"

        return {
            "diagnoses": diagnoses,
            "report": report_content
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Medical diagnosis failed: {str(e)}")


# 6.Edugpt
from wyge.prebuilt_agents.teaching_agent import teaching_agent_fun
from wyge.prebuilt_agents.generating_syllabus import generate_syllabus


async def start_learning(request: Request, topic: str, api_key: str) -> Dict[str, Any]:
    """
    Starts the learning process by generating a syllabus and initializing cache data.
    """
    try:
        if not topic or not topic.strip():
            raise HTTPException(status_code=400, detail="Topic is required")

        # Generate the syllabus
        syllabus = generate_syllabus(api_key, topic, "Focus on providing a clear learning path.")

        # Extract only the content if it's an AIMessage
        if hasattr(syllabus, "content"):
            syllabus = syllabus.content

        # Generate a unique cache key (using client IP + custom identifier)
        client_ip = request.client.host if request.client else "0.0.0.0"
        cache_key = f"edu_data_{client_ip}_{uuid.uuid4().hex[:8]}"

        # Store syllabus and topic in cache
        await FastAPICache.set(
            key=cache_key,
            value={
                "syllabus": syllabus,
                "current_topic": topic,
                "messages": []  # Initialize conversation history
            },
            expire=3600  # Cache for 1 hour
        )

        return {
            "topic": topic,
            "syllabus": syllabus,
            "cache_key": cache_key  # Return cache key to client for subsequent requests
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting learning: {str(e)}")


async def chat_with_agent(cache_key: str, user_input: str, api_key: str) -> Dict[str, Any]:
    """
    Handles user interaction with the teaching agent using cache-based session storage.
    """
    try:
        if not cache_key:
            raise HTTPException(status_code=400, detail="Cache key is required")
        if not user_input or not user_input.strip():
            raise HTTPException(status_code=400, detail="User input cannot be empty")

        # Retrieve cached data
        edu_data = await FastAPICache.get(key=cache_key)
        if not edu_data or "syllabus" not in edu_data or "current_topic" not in edu_data:
            raise HTTPException(status_code=400, detail="No training data found. Please start learning first.")

        # Recreate the teaching agent dynamically
        teaching_agent = teaching_agent_fun(api_key)

        # Retrieve messages from the cache
        conversation_history = edu_data.get("messages", [])

        # Rebuild past conversation
        for msg in conversation_history:
            if msg["role"] == "user":
                teaching_agent["add_user_message"](msg["content"])
            elif msg["role"] == "assistant":
                teaching_agent["add_ai_message"](msg["content"])

        # Add new user message
        teaching_agent["add_user_message"](user_input)

        # Generate AI response
        response = teaching_agent["generate_response"]()

        # Extract AI response content
        response_content = response.content if hasattr(response, "content") else str(response)

        # Update conversation history with JSON-safe data
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response_content})

        # Store updated conversation in cache
        edu_data["messages"] = conversation_history
        await FastAPICache.set(key=cache_key, value=edu_data, expire=3600)

        return {
            "user_message": user_input,
            "assistant_response": response_content,
            "cache_key": cache_key  # Return same cache key for subsequent requests
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat with agent: {str(e)}")


# 7.Medical Image Processing agent
from wyge.prebuilt_agents.medical_image_processing import MedicalImageAnalyzer


async def medical_image_analysis(api_key: str, uploaded_file: UploadFile) -> Dict[str, Any]:
    try:
        # Validate file extension
        allowed_extensions = {".jpg", ".jpeg", ".png"}
        file_extension = os.path.splitext(uploaded_file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload a JPG, JPEG, or PNG file."
            )

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # Read and save file content
            file_content = await uploaded_file.read()
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            # Analyze the image
            analyzer = MedicalImageAnalyzer(api_key=api_key)
            result = analyzer.analyze_image(tmp_file_path)
            simplified_explanation = analyzer.simplify_explanation(result)

            return {
                "result": result,
                "simplified_explanation": simplified_explanation
            }

        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing medical image: {str(e)}"
        )


# 8.Image answering
from wyge.prebuilt_agents.vqa import VisualQA


async def visual_question_answering(api_key: str, uploaded_file: UploadFile, question: str) -> Dict[str, Any]:
    try:
        # Validate file extension
        allowed_extensions = {".jpg", ".jpeg", ".png"}
        file_extension = os.path.splitext(uploaded_file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload a JPG, JPEG, or PNG file."
            )

        if not question or not question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # Read and save file content
            file_content = await uploaded_file.read()
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            # Initialize VQA agent and get answer
            vqa = VisualQA(api_key=api_key)
            response = vqa.ask(image_path=tmp_file_path, question=question.strip())

            return {"answer": response}

        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing visual question: {str(e)}"
        )


# 9,10 Blog Agents (urls +file)
from wyge.prebuilt_agents.social_media_agents import ResearchAgent, BlogAgent, LinkedInAgent


async def generate_blog_from_url(prompt: str, url: str, option: str, api_key: str) -> Dict[str, Any]:
    """
    Generates blog or LinkedIn post content from a website URL.
    """
    try:
        if not url:
            raise HTTPException(status_code=400, detail="Website URL is required")
        if option not in ['blog_post', 'linkedin_post']:
            raise HTTPException(status_code=400, detail="Invalid option specified")

        research_agent = ResearchAgent(api_key=api_key)

        if option == 'blog_post':
            blog_agent = BlogAgent(api_key=api_key)
            content = research_agent.research_website(prompt, url)
            blog_post = blog_agent.generate_blog(prompt, str(content))
            return {'content': blog_post}
        else:
            linkedin_agent = LinkedInAgent(api_key=api_key)
            content = research_agent.research_website(prompt, url)
            linkedin_post = linkedin_agent.generate_linkedin_post(prompt, str(content))
            return {'content': linkedin_post}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating content from URL: {str(e)}")


async def generate_blog_from_yt_url(prompt: str, url: str, option: str, api_key: str) -> Dict[str, Any]:
    """
    Generates blog or LinkedIn post content from a YouTube URL.
    """
    try:
        if not url:
            raise HTTPException(status_code=400, detail="YouTube URL is required")
        if option not in ['blog_post', 'linkedin_post']:
            raise HTTPException(status_code=400, detail="Invalid option specified")

        research_agent = ResearchAgent(api_key=api_key)
        content = research_agent.extract_transcript_from_yt_video(url)

        if option == 'blog_post':
            blog_agent = BlogAgent(api_key=api_key)
            blog_post = blog_agent.generate_blog(prompt, str(content))
            return {'content': blog_post}
        else:
            linkedin_agent = LinkedInAgent(api_key=api_key)
            linkedin_post = linkedin_agent.generate_linkedin_post(prompt, str(content))
            return {'content': linkedin_post}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating content from YouTube: {str(e)}")


async def generate_blog_from_file(prompt: str, file: UploadFile, option: str, api_key: str) -> Dict[str, Any]:
    """
    Generates blog or LinkedIn post content from an audio file.
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Audio file is required")
        if option not in ['blog_post', 'linkedin_post']:
            raise HTTPException(status_code=400, detail="Invalid option specified")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file_content = await file.read()
            tmp_file.write(file_content)
            temp_path = tmp_file.name

        try:
            research_agent = ResearchAgent(api_key=api_key)
            content = research_agent.extract_audio_transcript(temp_path)

            if option == 'blog_post':
                blog_agent = BlogAgent(api_key=api_key)
                blog_post = blog_agent.generate_blog(prompt, str(content))
                return {'content': blog_post}
            else:
                linkedin_agent = LinkedInAgent(api_key=api_key)
                linkedin_post = linkedin_agent.generate_linkedin_post(prompt, str(content))
                return {'content': linkedin_post}

        finally:
            # Cleanup temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating content from file: {str(e)}")


# 11.Mcq Generator
from wyge.prebuilt_agents.mcq_generator import MCQGeneratorAgent


async def mcq_generation(api_key: str, user_prompt: str) -> Dict[str, Any]:
    try:
        if not user_prompt or not user_prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty. Please provide a topic or content."
            )

        # Initialize the MCQ generator agent
        agent = MCQGeneratorAgent(api_key)
        mcq_set = agent.generate_mcq_set(user_prompt.strip())

        if not mcq_set:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate MCQs. Please try again with a different prompt."
            )

        return {"answer": mcq_set}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating MCQs: {str(e)}"
        )


# 12.Synthetic Data Generator
from wyge.prebuilt_agents.synthetic_data_generator import generate_synthetic_data, generate_data_from_text, \
    fill_missing_data_in_chunk


async def handle_synthetic_data_for_new_data(user_prompt: str, openai_api_key: str) -> Dict[str, Any]:
    try:
        # Extract number of rows from the prompt
        num_rows = extract_num_rows_from_prompt(user_prompt)
        if num_rows is None:
            raise HTTPException(
                status_code=400,
                detail="Number of rows or records not found in the prompt."
            )

        # Extract column names from the prompt
        column_names = extract_columns_from_prompt(user_prompt)
        if not column_names:
            raise HTTPException(
                status_code=400,
                detail="No field names found in the prompt."
            )

        # Generate synthetic data
        generated_df = generate_data_from_text(
            openai_api_key,
            user_prompt,
            column_names,
            num_rows=num_rows
        )

        # Convert to CSV
        csv_data = generated_df.to_csv(index=False)
        return {"data": csv_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating new data: {str(e)}"
        )


async def handle_synthetic_data_from_excel(
        file: UploadFile,
        openai_api_key: str,
        user_prompt: str
) -> Dict[str, Any]:
    try:
        # Validate file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".xlsx", ".csv"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload an Excel or CSV file."
            )

        # Read file content
        file_content = await file.read()

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            # Read original data
            if file_extension == ".xlsx":
                original_df = pd.read_excel(tmp_file_path)
            else:
                original_df = pd.read_csv(tmp_file_path)

            # Generate additional data
            num_rows = extract_num_rows_from_prompt(user_prompt)
            generated_df = generate_synthetic_data(
                openai_api_key,
                tmp_file_path,
                num_rows
            )

            # Combine and return
            combined_df = pd.concat([original_df, generated_df], ignore_index=True)
            return {"data": combined_df.to_csv(index=False)}

        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extending data: {str(e)}"
        )


async def handle_fill_missing_data(
        file: UploadFile,
        openai_api_key: str
) -> Dict[str, Any]:
    try:
        # Validate file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".xlsx", ".csv"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload an Excel or CSV file."
            )

        # Read file content
        file_content = await file.read()

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            # Read and process data
            if file_extension == ".xlsx":
                original_df = pd.read_excel(tmp_file_path)
            else:
                original_df = pd.read_csv(tmp_file_path)

            filled_df = fill_missing_data_in_chunk(openai_api_key, tmp_file_path)
            return {"data": filled_df.to_csv(index=False)}

        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error filling missing data: {str(e)}"
        )


# Helper functions (unchanged from original)
def extract_num_rows_from_prompt(user_prompt: str) -> Optional[int]:
    match = re.search(r'(\d+)\s+(rows|records)', user_prompt, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_columns_from_prompt(user_prompt: str) -> list:
    match = re.search(
        r'(field names|column names|fields|columns|field_names|column_names):\s*([a-zA-Z0-9_,\s\.]+)',
        user_prompt,
        re.IGNORECASE
    )
    if not match:
        return []

    raw_columns = match.group(2).split(',')
    formatted_columns = [
        re.sub(r'[^a-zA-Z0-9]', '_', col.strip()).lower()
        for col in raw_columns
    ]
    return list(dict.fromkeys(filter(bool, formatted_columns)))
