# main.py
import base64
import json
import datetime
import os
import io
import sys
import markdown
import pandas as pd
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Optional
import openai
from openai import OpenAI

# Load environment variables
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Initialize FastAPI app
app = FastAPI(
    title="Integrated API Services",
    description="Combines financial analysis, image generation, and SLA breach analysis",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the media folder
MEDIA_ROOT = 'media'
if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==============================================
# Financial Analysis Endpoints
# ==============================================

class QueryRequest(BaseModel):
    prompt: str


financial_prompt_template = """
You are an expert financial analyst providing real-time insights into company performance. Analyze the financial data of [Company Name] using the latest available information from reliable sources, including MarketScreener, Bloomberg, Mint, Wikipedia, and company filings.
Ensure the response follows this JSON format:
{
  "title": "[Suitable title with Company Name]",
  "summary": "Brief overview of financial performance of [Company Name]",
  "paragraphs": [
    {
      "content": "Detailed analysis including revenue,statistics, net profit, EPS, and growth of [Company Name]"
    },
    {
      "content": "Comparison with competitors and market trends of [Company Name]."
    },
    {
      "content": "Predicted growth areas and risks based on data ."
    }
  ]
}

Requirements:
-Do not include any '**' or '##" or anything in the final response.just give the plain text as the response.
-'Summary' should be present for 3-4 lines.
- The number of paragraphs will be 6-7 only.
-'Context' should be within 5-6 lines and the very important content should be there.
-'title' should be inserted based on the user query only.The title must be unique.
- All the content and the summary must and should collect from the Sources such as "Wikipedia" or "Company Press releases" or "each metric from reliable data" only.
"""


@app.post("/analyze")
def analyze_financial_query(request: QueryRequest):
    """Processes user query and returns structured financial response."""
    if not request.prompt:
        return {"error": "Query cannot be empty."}

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": financial_prompt_template},
                {"role": "user", "content": request.prompt}
            ],
            temperature=0.5,
            max_tokens=1200
        )

        raw_response = response.choices[0].message.content.strip()
        structured_data = json.loads(raw_response)
        return {"query": request.prompt, "response": structured_data}

    except json.JSONDecodeError:
        return {"error": "Failed to parse response as JSON."}
    except Exception as e:
        return {"error": str(e)}


# ==============================================
# Image Generation Endpoints
# ==============================================

class ImageRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"  # Default image size


def generate_image(prompt: str, size: str):
    """Calls OpenAI's DALL·E to generate an image based on the prompt."""
    valid_sizes = ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    if size not in valid_sizes:
        size = "1024x1024"  # Fallback to default

    response = client.images.generate(
        model="dall-e-3",
        prompt=f"Realistic illustration of images with rich details, lifelike characters, and a dynamic atmosphere for {prompt}",
        n=1,
        size=size
    )
    return response.data[0].url


def convert_image_to_base64(image_url: str):
    """Downloads the image and converts it to base64 encoding."""
    response = requests.get(image_url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    return None


@app.post("/image_generate")
def generate_image_api(request: ImageRequest):
    """Processes user prompt and returns an AI-generated image."""
    try:
        image_url = generate_image(request.prompt, request.size)
        image_base64 = convert_image_to_base64(image_url)
        if image_base64:
            return {"query": request.prompt, "image_base64": image_base64}
        return {"error": "Failed to convert image to base64."}
    except Exception as e:
        return {"error": str(e)}


# ==============================================
# SLA Analysis Endpoints
# ==============================================

def preprocess_data(csv_data):
    """Preprocesses the SLA data."""
    csv_data['Historical Status - Change Date'] = pd.to_datetime(
        csv_data['Historical Status - Change Date'], dayfirst=True, errors='coerce'
    )

    csv_data['Historical Status - Change Time'] = csv_data['Historical Status - Change Time'].astype(str).str.zfill(6)
    csv_data['Historical Status - Change Time'] = pd.to_datetime(
        csv_data['Historical Status - Change Time'], format='%H%M%S', errors='coerce'
    ).dt.time

    csv_data['Change Datetime'] = csv_data.apply(
        lambda row: datetime.datetime.combine(row['Historical Status - Change Date'],
                                              row['Historical Status - Change Time'])
        if pd.notnull(row['Historical Status - Change Date']) and pd.notnull(row['Historical Status - Change Time'])
        else pd.NaT,
        axis=1
    )

    grouped_data = csv_data.groupby(csv_data['Request - ID'])
    sorted_groups = []
    for _, group in grouped_data:
        group = group.sort_values(by=['Historical Status - Change Date', 'Change Datetime'])
        sorted_groups.append(group)

    return pd.concat(sorted_groups)


def calculate_working_hours(start, end):
    """Calculate working hours between two datetimes (2 PM to 11 PM, weekdays only)"""
    if not isinstance(start, datetime.datetime) or not isinstance(end, datetime.datetime):
        return 0.0

    working_hours_start = datetime.time(14, 0)  # 2 PM
    working_hours_end = datetime.time(23, 0)  # 11 PM

    if start >= end:
        return 0.0

    total_hours = 0.0
    current_date = start.date()
    end_date = end.date()

    while current_date <= end_date:
        if current_date.weekday() >= 5:  # Skip weekends
            current_date += datetime.timedelta(days=1)
            continue

        day_start = datetime.datetime.combine(current_date, working_hours_start)
        day_end = datetime.datetime.combine(current_date, working_hours_end)

        interval_start = max(start, day_start)
        interval_end = min(end, day_end)

        if interval_start < interval_end:
            delta = interval_end - interval_start
            total_hours += delta.total_seconds() / 3600

        current_date += datetime.timedelta(days=1)

    return round(total_hours, 2)


def calculate_time_differences(csv_data):
    """Calculate time differences between status changes."""
    csv_data['Change'] = 0.0
    csv_data['Change Datetime'] = pd.to_datetime(
        csv_data['Change Datetime'],
        errors='coerce',
        dayfirst=True
    )

    valid_rows = csv_data['Change Datetime'].notna()
    if not valid_rows.all():
        print(f"Warning: Dropped {len(csv_data) - valid_rows.sum()} rows with invalid datetime values")
    csv_data = csv_data[valid_rows].copy()

    if not pd.api.types.is_datetime64_any_dtype(csv_data['Change Datetime']):
        raise ValueError("Failed to convert 'Change Datetime' to proper datetime format")

    for request_id, group in csv_data.groupby('Request - ID'):
        group = group.sort_values('Change Datetime')
        previous_time = None

        for index, row in group.iterrows():
            current_time = row['Change Datetime']

            if not isinstance(current_time, datetime.datetime):
                csv_data.at[index, 'Change'] = 0.0
                continue

            is_weekend = current_time.weekday() >= 5

            if previous_time is None:
                if not is_weekend and current_time.time() >= datetime.time(14, 0):
                    start_of_day = current_time.replace(
                        hour=14, minute=0, second=0, microsecond=0
                    )
                    work_hours = calculate_working_hours(start_of_day, current_time)
                else:
                    work_hours = 0.0
            else:
                work_hours = calculate_working_hours(previous_time, current_time)

            csv_data.at[index, 'Change'] = work_hours
            previous_time = current_time

    allowed_transitions = {
        ("Forwarded", "Assigned"),
        ("Forwarded", "Work in progress"),
        ("Assigned", "Work in progress"),
        ("Work in progress", "Suspended"),
        ("Work in progress", "Solved"),
        #("Suspended", "Solved"),
        ("Forwarded", "Suspended")
    }

    transition_pairs = csv_data[
        ['Historical Status - Status From', 'Historical Status - Status To']
    ].apply(tuple, axis=1)

    return csv_data[transition_pairs.isin(allowed_transitions)].copy()


def calculate_sla_breach(csv_data):
    """Calculate SLA breach status."""
    sla_mapping = {"P1 - Critical": 4, "P2 - High": 8, "P3 - Normal": 45, "P4 - Low": 90}
    csv_data['SLA Hours'] = csv_data['Request - Priority Description'].map(sla_mapping)
    csv_data['Total Elapsed Time'] = csv_data.groupby('Request - ID')['Change'].transform('sum').round(2)
    csv_data['Time_to_breach'] = csv_data['SLA Hours'] - csv_data['Total Elapsed Time'].round(2)

    csv_data['Time_to_breach'] = np.where(
        csv_data['Total Elapsed Time'] > csv_data['SLA Hours'],
        0,
        csv_data['Time_to_breach']
    )

    csv_data['Breached'] = np.where(csv_data['Total Elapsed Time'] > csv_data['SLA Hours'], 'Yes', 'No')
    csv_data['Final_Status'] = csv_data.groupby('Request - ID')['Historical Status - Status To'].transform('last')
    return csv_data


def generate_report(csv_data):
    """Generate the final report."""
    ticket_groups = {}

    for _, row in csv_data.iterrows():
        ticket_id = row['Request - ID']
        status_from = row['Historical Status - Status From']
        status_to = row['Historical Status - Status To']
        date = pd.to_datetime(f"{row['Historical Status - Change Date']} {row['Historical Status - Change Time']}",
                              errors='coerce')

        if ticket_id not in ticket_groups:
            ticket_groups[ticket_id] = []

        ticket_groups[ticket_id].append({
            'row': row,
            'date': date,
            'statusFrom': status_from,
            'statusTo': status_to
        })

    filtered_records = []
    for records in ticket_groups.values():
        if records:
            filtered_records.append(records[-1]['row'])

    filtered_data = pd.DataFrame(filtered_records)

    report_data = filtered_data[[
        'Request - ID', 'Request - Priority Description', 'Request - Resource Assigned To - Name',
        'SLA Hours', 'Total Elapsed Time', 'Time_to_breach', 'Final_Status', 'Breached'
    ]].drop_duplicates()

    report_data.rename(columns={
        'Request - ID': 'Ticket',
        'Request - Priority Description': 'Priority',
        'Request - Resource Assigned To - Name': 'Assigned To',
        'SLA Hours': 'Allowed Duration(in Hours)',
        'Total Elapsed Time': 'Total Elapsed Time(in Hours)',
        'Time_to_breach': 'Time to Breach(in Hours)',
        'Final_Status': 'Status',
        'Breached': 'Breached'
    }, inplace=True)

    return report_data


def save_report_to_media(report_data):
    """Save report to media folder."""
    report_file_path = os.path.join(MEDIA_ROOT, 'final_report.csv')
    report_data.to_csv(report_file_path, index=False)
    return report_file_path


def generate_code(prompt_eng):
    """Generate Python code for SLA analysis."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    all_text = response.choices[0].message.content.strip()
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    return code


def execute_py_code(code, df):
    """Execute generated Python code."""
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
                output = str(output)
    except Exception as e:
        output = f"Error executing code: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__

    return output


def generate_response(prompt_eng):
    """Generate response for SLA queries."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing actionable insights."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    return response.choices[0].message.content.strip()


def markdown_to_html(md_text):
    """Convert markdown to HTML."""
    html_content = markdown.markdown(md_text)
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        tag['style'] = "color: blue;"
    return str(soup)


@app.post("/sla_processing")
async def process_sla_file(file: UploadFile = File(...)):
    """Process SLA CSV file and generate report."""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

        file_path = os.path.join(MEDIA_ROOT, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        csv_data = pd.read_csv(file_path)
        sorted_data = preprocess_data(csv_data)
        csv_data = calculate_time_differences(sorted_data)
        csv_data = calculate_sla_breach(csv_data)
        report_data = generate_report(csv_data)
        report_file_path = save_report_to_media(report_data)

        return JSONResponse({
            "message": "Report generated successfully.",
            "report_path": report_file_path,
            "report_data": report_data.to_dict(orient='records')
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/sla_query")
async def sla_query(query: str = Form(...)):
    """Handle SLA-related queries."""
    try:
        greetings = {"hi", "hello", "hey", "greetings"}

        if query.lower() in greetings:
            greeting_response = generate_response("Respond to the user greeting in a friendly and engaging manner.")
            return JSONResponse({"answer": markdown_to_html(greeting_response)})

        csv_file_path = os.path.join(MEDIA_ROOT, 'final_report.csv')
        if not os.path.exists(csv_file_path):
            raise HTTPException(status_code=400, detail="CSV file not found. Please upload the file first.")

        df = pd.read_csv(csv_file_path)
        metadata_str = ", ".join(df.columns.tolist())

        prompt_eng = f"""
            You are a Python expert specializing in data preprocessing. Your task is to answer user queries strictly based on the dataset `{csv_file_path}`. Follow these strict rules:

            1. Strict Dataset Constraints:
                - You can only refer to the columns and data present in the CSV.
                - Do not make any assumptions or external calculations.
                - Available columns: {metadata_str}.
                - Dataset preview (first 10 rows):  
                  {df.head(10)}

            2. Rules for Query Execution:
                - Perform operations directly on the dataset.
                - Do not assume missing data unless explicitly mentioned.
                - No implicit conversions (e.g., do not convert hours to minutes unless specified).
                - Only return results filtered exactly as per the query.

            3. Handling Ticket Queries:
                - If the query is about breached tickets, only consider rows where 'Breached' == "Yes".
                - For queries involving ticket lists, only return the following columns:
                    - Ticket  
                    - Priority  
                    - Assigned To  
                    - Allowed Duration  
                    - Total Elapsed Time  
                    - Time to Breach  
                    - Status  
                    - Breached  

            4. Code Structure Guidelines:
                - Provide only Python code, no explanations.
                - Ensure the code:
                    - Loads `{csv_file_path}` using pandas.
                    - Filters and processes data based on the query.
                    - Uses comments for readability.

            5. Tabular Output for React Compatibility:
                - Format the output as an HTML table for clarity.
                - Use proper `<table>`, `<thead>`, `<tbody>`, `<tr>`, and `<td>` tags.
                - Ensure the table structure is well-formed.

            6. Download Handling:
                - Only generate a downloadable CSV file if explicitly requested.
                - If requested, provide a script that saves the filtered data as `filtered_data_<timestamp>.csv`.

            7. Strict Query Handling:
                - If the query is unclear, return "Invalid query: Please clarify your request."
                - Do not generate responses based on assumptions.

            User Query: {query}
            """

        code = generate_code(prompt_eng)
        result = execute_py_code(code, df)
        return JSONResponse({"answer": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/sla/download")
async def download_sla_report():
    """Download the SLA report."""
    report_path = os.path.join(MEDIA_ROOT, 'final_report.csv')
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, filename="final_report.csv")


# ==============================================
# Main Application
# ==============================================

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
