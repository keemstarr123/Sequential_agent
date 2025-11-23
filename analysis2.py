import prompt
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from googleapiclient.discovery import build
import pprint
import asyncio
import httpx
from bs4 import BeautifulSoup
from langchain_core.exceptions import OutputParserException
import time
from google import genai
import cv2
from http import HTTPStatus
import requests
from moviepy import concatenate_videoclips, VideoFileClip 
from io import BytesIO
import tempfile
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from typing import Dict, Any, Optional
import os
import requests
from datetime import timedelta
import datetime
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient, BlobClient, BlobSasPermissions, generate_blob_sas
from google.cloud import firestore
from google.oauth2 import service_account
import re


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # or list of allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Clusters(BaseModel):
    pattern_output: prompt.PatternAnalystOutput
    fourc_output: prompt.FourCAnalystOutput
    marketing_strategist_output: prompt.strategistOutput
    number_of_customers: int = 10
    definition: str = ""
    video_link:list[str] = []

class ImplementationInput(BaseModel):
    analysis_id: str
    solution_idx: int
    

# Store progress in firebase
def create_run_doc() -> str:
    doc_ref = db.collection(RUNS_COLLECTION).document()
    doc_ref.set({
        "status": JobStatus.QUEUED.value,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "steps": {
            "DataBricks": {"status": JobStatus.QUEUED.value},
            "patternAnalyst": {"status": JobStatus.QUEUED.value},
            "fourCAnalyst": {"status": JobStatus.QUEUED.value},
            "marketingStrategist": {"status": JobStatus.QUEUED.value},
            "videoMakingAgent": {"status": JobStatus.QUEUED.value},
        },
        "error": None,
        "analysisRef": None,       # will fill later
    })
    return doc_ref.id


def update_run(run_id: str, data: dict):
    db.collection(RUNS_COLLECTION).document(run_id).update({
        **data,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })


def update_step(run_id: str, step_name: str, status: JobStatus, extra: Optional[dict] = None):
    update = {
        f"steps.{step_name}.status": status.value,
        f"steps.{step_name}.finishedAt": firestore.SERVER_TIMESTAMP
            if status in (JobStatus.COMPLETED, JobStatus.FAILED) else firestore.SERVER_TIMESTAMP,
    }
    if extra:
        for k, v in extra.items():
            update[f"steps.{step_name}.{k}"] = v

    update_run(run_id, update)

tool_call_state = {"count": 0, "limit": 10}


load_dotenv()

gcp_key = os.getenv("GOOGLE_API_KEY")
dashscope_key = os.getenv("DASHSCOPE_API_KEY")
cse_id = os.getenv("GOOGLE_CSE_ID")
client = genai.Client(api_key=gcp_key)
openai_client =  OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
sas_token = os.getenv("SAS_TOKEN")
json_data = json.loads(os.getenv("FIREBASE_ACC"))
with open("service-account.json", "w") as f:
    json.dump(json_data, f)

# ----------------------------
# FIREBASE
# ----------------------------
creds = service_account.Credentials.from_service_account_file("service-account.json")
db = firestore.Client(credentials=creds, project="foresee-80b0c")

RUNS_COLLECTION = "runs"       # orchestration/progress
ANALYSES_COLLECTION = "analyses"
IMPLEMENTATION_COLLECTION = "implementations"

# ----------------------------
# Destination ADLS (your lake)
# ----------------------------
DEST_ACCOUNT = "fastlakecentral"
DEST_CONTAINER = "raw"
DEST_KEY = os.getenv("DEST_ADLS_ACCOUNT_KEY")

DEST_AZURE_BASE = "clientdata"

# ----------------------------
# Databricks Configuration
# ----------------------------
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_JOB_ID = os.getenv("DATABRICKS_JOB_ID")


class Cluster:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.num = get_num(self.df)

class RunCheckRequest(BaseModel):
    run_id: str

def extract_json_block(text: str) -> str:
    # If there is a ```json ``` fenced block, prefer that
    fence_match = re.search(r"```json(.*)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    
    # Fallback: grab substring between first { and last }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last+1].strip()
    
    # if nothing plausible
    raise OutputParserException("No JSON object found in LLM output")




def get_num(df):
    return df.shape[0]

async def fetch_pages(url):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            body = soup.find("body")
            if body:
                return body.get_text(separator="\n", strip=True)
            else:
                return f"[No <body> tag found for {url}]"
        except Exception as e:
            return f"[Error fetching {url}]: {str(e)}"
    

async def main(urls):
    tasks = [fetch_pages(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results


@tool
def search_web(query:str, depth=0):
    """
    Searches the internet in real-time to retrieve information not present in the LLM's training data.

    This tool:
    - Uses a live search engine to find relevant web pages for the given query.
    - Fetches and parses the HTML content of each result.
    - Returns the raw or structured content (e.g., page body text) from those pages.

    Useful for answering up-to-date questions about recent events, facts, or topics the LLM may not know.

    Args:
        query (str): The user's search question or topic.
        depth (int): The number of queries made so far in this session.

    Returns:
        dict: A dictionary containing search results with their URLs and page contents.
    """
    if depth >= tool_call_state["limit"]:
        print(f"⚠️ Recursion limit reached ({tool_call_state['limit']}). Stopping.")
        return {"error": "Recursion limit exceeded. Summarize your findings now."}
    service = build(
        "customsearch", "v1", developerKey=gcp_key
    )

    res = (
        service.cse()
        .list(
            q=query,
            cx=cse_id,
        )
        .execute()
    )
    urls = [i['link'] for i in res['items']]
    contents = asyncio.run(main(urls))
    res['items'] =[
    {**j, "contents": contents[i]}
    for i, j in enumerate(res["items"])
    ]
    return res

@tool
def get_record(pd_statement:str, depth=0):
        """
        Execute a pandas expression string using the provided DataFrame.
        To access the DataFrame, use 'cluster.df' within the pd_statement.
        Args:
            pd_statement (str): A string containing a valid Pandas expression.
            depth (int): The number of queries made so far in this session.
        Returns:
            Any: The result of the evaluated Pandas expression.

        Example: get_record("cluster.df[:200]") 
        """
        if depth >= tool_call_state["limit"]:
            print(f"⚠️ Recursion limit reached ({tool_call_state['limit']}). Stopping.")
            return {"error": "Recursion limit exceeded. Summarize your findings now."}
        try:
            return eval(pd_statement) 
        except RecursionError:
            return "⚠️ RecursionError detected. Halting to prevent infinite loop."   
        except Exception as e:
            return str(e)
        
def trigger_databricks_job(job_id: str):
    if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, job_id]):
        print("Skipping Databricks trigger: Missing configuration.")
        return

    url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/run-now"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    payload = {"job_id": job_id}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        run_id = response.json().get('run_id')
        print(f"Triggered Databricks job {job_id}. Run ID: {run_id}")
        return run_id
    except Exception as e:
        print(f"Failed to trigger Databricks job: {e}")


def adls_path(*parts: str) -> str:
    # normalise path segments
    clean = [p.strip("/").replace("\\", "/") for p in parts if p]
    return "/".join(clean)

class AzureUrlIngestRequest(BaseModel):
    source_url: str            # SAS URL for container or folder
    max_files: int = 1000
    skip_zero_length: bool = True

class AzureUrlIngestRequest2(BaseModel):
    source_url: str            # SAS URL for container or folder
    id: str       
    max_files: int = 1000
    skip_zero_length: bool = True

def _copy_from_url(src_url: str, dest_path: str):
    """
    Copies a file from a public/SAS Azure Blob URL into your destination Data Lake.
    """
    # initialize destination blob client
    dest_blob = BlobClient(
        account_url=f"https://{DEST_ACCOUNT}.blob.core.windows.net",
        container_name=DEST_CONTAINER,
        blob_name=dest_path,
        credential=DEST_KEY,   # your Data Lake key
    )

    # copy directly from the SAS source
    dest_blob.start_copy_from_url(src_url)

def ingest_azure_adls_from_url(req: AzureUrlIngestRequest):
    # --- Parse the SAS URL ---
    parsed = urlparse(req.source_url)
    if not parsed.scheme.startswith("http"):
        raise HTTPException(status_code=400, detail="source_url must be an https:// URL")

    host = parsed.hostname or ""
    if ".blob.core.windows.net" not in host:
        raise HTTPException(status_code=400, detail="source_url must be an Azure Blob endpoint")

    account = host.split(".")[0]          # e.g. clientlakedev
    path = parsed.path.lstrip("/")        # e.g. "exports/2025" or "exports"
    if not path:
        raise HTTPException(status_code=400, detail="URL must include container name in the path")

    parts = path.split("/", 1)
    container = parts[0]                   # "exports"
    prefix = parts[1] if len(parts) > 1 else ""   # "2025" or ""

    # normalise prefix "2025" -> "2025/"
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    sas = "?" + (parsed.query or "")
    if "sig=" not in sas:
        raise HTTPException(status_code=400, detail="source_url must contain a SAS token (sig=...)")

    # --- Source blob client using SAS ---
    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=sas
    )
    container_client = service.get_container_client(container)

    # --- FIXED DESTINATION ---
    dest_base = DEST_AZURE_BASE

    copied, skipped = 0, 0

    # --- List & copy blobs ---
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if copied >= req.max_files:
            break

        size = getattr(blob, "size", None)
        if req.skip_zero_length and (size is None or size == 0):
            # skip folder markers / empty blobs
            skipped += 1
            continue

        blob_name = blob.name               # e.g. "2025/file1.xlsx"
        filename = blob_name.rstrip("/").split("/")[-1]

        src_url = f"https://{account}.blob.core.windows.net/{container}/{blob_name}{sas}"
        dest_path = adls_path(dest_base, filename)

        _copy_from_url(src_url, dest_path)
        copied += 1

    # --- Trigger Databricks Job ---
    if copied > 0 and DATABRICKS_JOB_ID:
        run_id = trigger_databricks_job(DATABRICKS_JOB_ID)

    return {
        "status": "ok",
        "run_id": run_id if copied > 0 and DATABRICKS_JOB_ID else None,
        "mode": "fixed_dest",
        "source_account": account,
        "source_container": container,
        "source_prefix": prefix,
        "dest_base": dest_base,
        "copied": copied,
        "skipped": skipped
    }

def concat_videos_moviepy(
    paths,
    out_path="combined_output.mp4",
    target_fps=None,       # e.g. 25 or 30
    target_size=None       # e.g. (1080, 1920) or (1280, 720)
):
    if not paths:
        raise ValueError("No video paths provided")

    clips = []

    for p in paths:
        if not os.path.exists(p):
            print(f"⚠️ Skipping (file not found): {p}")
            continue

        try:
            clip = VideoFileClip(p)
        except Exception as e:
            print(f"⚠️ Skipping (cannot open): {p} ({e})")
            continue

        # Resize if requested
        if target_size is not None:
            # target_size = (width, height)
            clip = clip.resize(newsize=target_size)

        # Normalize FPS if requested (MoviePy v2 uses with_fps)
        if target_fps is not None:
            clip = clip.with_fps(target_fps)

        clips.append(clip)

    if not clips:
        raise RuntimeError("No valid clips to concatenate")

    # Concatenate video + audio together – MoviePy keeps them aligned
    final_clip = concatenate_videoclips(clips, method="compose")

    # Choose FPS: use target_fps if set, else keep first clip's FPS
    fps_to_use = target_fps or clips[0].fps or 24

    final_clip.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps_to_use
    )

    # Cleanup resources
    final_clip.close()
    for c in clips:
        c.close()

    print(f"✅ Combined video with audio saved as {out_path}")


def wait_for_video_to_finish(video_id: str, poll_interval: int = 5, timeout: int = 600):
    """
    Poll Sora until the video job is completed or failed.
    Returns the final job object.
    """
    elapsed = 0
    while elapsed < timeout:
        job = openai_client.videos.retrieve(video_id)
        status = job.status
        progress = getattr(job, "progress", None)
        print(f"Status: {status}", f"- {progress}%" if progress is not None else "")

        if status == "completed":
            return job
        if status == "failed":
            msg = getattr(getattr(job, "error", None), "message", "Unknown error")
            raise RuntimeError(f"Failed to generate video: {msg}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise RuntimeError("Polling timed out before completion")


def download_video(video_id: str, out_path) -> str:
    """
    Download the generated video bytes and save as MP4.
    Returns the local file path.
    """
    if out_path is None:
        out_path = f"{video_id}.mp4"

    # Ensure .mp4 extension
    if not out_path.lower().endswith(".mp4"):
        out_path = out_path + ".mp4"

    resp =  openai_client.videos.download_content(video_id=video_id)
    video_bytes = resp.read()

    with open(out_path, "wb") as f:
        f.write(video_bytes)

    print(f"✅ Saved video to {out_path}")
    return out_path

def upload_video_to_blob(local_file_path, blob_name):
    blob = BlobClient(
        account_url=f"https://{DEST_ACCOUNT}.blob.core.windows.net",
        container_name= DEST_CONTAINER,
        blob_name=blob_name,
        credential=DEST_KEY
    )
    with open(local_file_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)

    sas_token = generate_blob_sas(
        account_name=DEST_ACCOUNT,           # STORAGE ACCOUNT NAME
        container_name=DEST_CONTAINER,       # SAME CONTAINER
        blob_name=blob_name,                 # SAME FILE
        account_key=DEST_KEY,                # ACCOUNT KEY
        permission=BlobSasPermissions(read=True),
        expiry=datetime.datetime.now(datetime.timezone.utc) + timedelta(hours=500)
    )

    sas_url = f"https://{DEST_ACCOUNT}.blob.core.windows.net/{DEST_CONTAINER}/{blob_name}?{sas_token}"  # or generate SAS token
    return sas_url

def generate_sora2_video(
    prompt: str,
    model: str = "sora-2",          # or "sora-2-pro"
    size: str = "1280x720",         # e.g. "720x1280" for vertical
    seconds: int = 8,               # 4, 8, 12… depending on your access
    out_path: Optional[str] = None,
) -> str:
    """
    High-level helper:
    1) create a Sora 2 video job
    2) wait for completion
    3) download as MP4
    Returns local video file path.
    """
    video_job =  openai_client.videos.create(
        model=model,
        prompt=prompt,
        size=size,
        seconds=seconds,
    )

    video_id = video_job.id
    print(f"🚀 Started Sora 2 job: {video_id}")

    # Wait until it's done
    wait_for_video_to_finish(video_id)

    # Download the file
    return download_video(video_id, out_path)

from concurrent.futures import ThreadPoolExecutor, as_completed

def video_making_agent(template: str, context: str, parser: PydanticOutputParser, file_name: str):
    full_prompt = template + f"\n\nCONTEXT:\n{context}\n\n"

    llm_resp = llm.invoke(full_prompt)
    raw_text = llm_resp.content[0]['text'] if hasattr(llm_resp, "content") else str(llm_resp)
    result = parser.parse(raw_text)
    print("📜 Storyboard result:", result)

    data = result.model_dump()  # e.g. {"Video_1": "...", "Video_2": "..."}
    items = list(data.items())  # keep order
    video_paths_by_name = {}

    def generate_clip(name: str, prompt_text: str) -> tuple[str, str]:
        print(f"🎥 Generating clip for {name} ...")
        out_name = f"{name}.mp4"
        path = generate_sora2_video(
            prompt=prompt_text,
            model="sora-2",
            size="1280x720",
            seconds='8',
            out_path=out_name,
        )
        return name, path

    # 🔁 Run Sora generations in parallel threads
    max_workers = 2  # tweak for your rate limits
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_clip, name, prompt_text)
            for name, prompt_text in items
        ]

        for fut in as_completed(futures):
            name, path = fut.result()
            video_paths_by_name[name] = path
            print(f"✅ Finished {name}: {path}")

    # Preserve original ordering (Video_1, Video_2, ...)
    video_paths = [video_paths_by_name[name] for name, _ in items]

    print("🧵 Concatenating final video...")
    concat_videos_moviepy(video_paths, out_path="combined_output.mp4", target_fps=24)

    url = upload_video_to_blob("combined_output.mp4", file_name)

    return url


#def implementation_planner(template, context, parser: PydanticOutputParser):


           

pattern_parser = PydanticOutputParser(pydantic_object=prompt.Pattern_Analyst_Config.format)
fourc_parser = PydanticOutputParser(pydantic_object=prompt.FourC_Analyst_Config.format)
strategist_parser = PydanticOutputParser(pydantic_object=prompt.Marketing_Strategist_Config.format) 
video_making_agent_parser = PydanticOutputParser(pydantic_object=prompt.Video_Making_Agent_Config.format)
claude_api_key = os.getenv("CLAUDE_API")
cluster = Cluster("./data/cluster_3.csv")
llm = ChatGoogleGenerativeAI(google_api_key=gcp_key, model="gemini-3-pro-preview", temperature=0.2)



Researching_Tool = create_agent(
    llm,
    tools=[get_record],
    system_prompt = ""
)

Pattern_Analyst = create_agent(
    #Loop agent sequence
    llm,
    tools=[get_record],
    system_prompt = prompt.Pattern_Analyst_Config.prompt_template,

)

FourC_analyst = create_agent(
    llm,
    tools=[get_record, search_web],
    system_prompt = prompt.FourC_Analyst_Config.prompt_template,
)


Marketing_strategist = create_agent(
    llm,
    tools=[search_web],
    system_prompt = prompt.Marketing_Strategist_Config.prompt_template,
)


import time
start = time.time()
print("Analyzing customer cluster data...")

def build_analysis_doc(
    cluster_number: int,
    cluster: Clusters,
    total_loss: float,
) -> dict:
    """
    Convert a Clusters object into the Firestore document shape you described.
    """
    p = cluster.pattern_output
    f = cluster.fourc_output
    s = cluster.marketing_strategist_output

    doc = {
        "cluster_number": cluster_number,
        # rename this however you decide in Firestore
        "potential_churn_customers": cluster.number_of_customers,
        "total_loss": total_loss,
        "clusters": [
            {
                # you can plug in a specific customer_no if you have it,
                # or leave "" and later extend your model
                "customer_no": "",
                "definition": cluster.definition or p.Definition,
                "common_characteristic": {
                    "Behavior": p.Behavior,
                    "Device_Usage": p.Device_Usage,
                    "Engagement": p.Engagement,
                    "Psychographic": p.Psychographic,
                },
                "four_c_analysis": {
                    "customer": f.Customer,
                    "cost": f.Cost,
                    "convenience": f.Convenience,
                    "communication": f.Communication,
                },
                "strategy": {
                    "title": s.Title,
                    "strategy_summary": s.Strategy_Summary,
                    "benefits": s.Benefits,
                    "strategic_pillars": s.Strategic_Pillars,
                    "tactical_execution": s.Tactical_Execution,
                    "budget_allocation": s.Budget_Allocation,
                    "financial_projection": s.Financial_Projection,
                    "expected_impact": s.Expected_Impact,
                    # store as array of strings in Firestore
                    "video_link": cluster.video_link,
                },
            }
        ],
    }
    return doc

def save_analysis_to_firestore(
    cluster_number: int,
    cluster: Clusters,
    total_loss: float,
) -> str:
    doc_data = build_analysis_doc(cluster_number, cluster, total_loss)
    ref = db.collection(ANALYSES_COLLECTION).document()
    ref.set(doc_data)
    return ref.id

def get_databricks_run_status(run_id: int | str) -> dict:
    """
    Check the current status of a Databricks job run.
    Returns a small dict like:
    {
        "life_cycle_state": "TERMINATED",
        "result_state": "SUCCESS",
        "state_message": "..."
    }
    """
    if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, run_id]):
        raise RuntimeError("Missing Databricks configuration or run_id")

    url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    params = {"run_id": int(run_id)}

    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    state = data.get("state", {}) or {}
    return {
        "life_cycle_state": state.get("life_cycle_state"),
        "result_state": state.get("result_state"),
        "state_message": state.get("state_message"),
        "raw": state,
    }

def run_analyze_cluster_job(run_id: str, req: AzureUrlIngestRequest2):
    try:
        if req.id == 1 or req.id == 0:
            analysis_id = "EtKEJA3lpCjEOz289wdN" if req.id == 1 else "hNXinSgZn1YBr9IkCJVF"
            update_run(run_id, {"status": JobStatus.RUNNING.value})
            time.sleep(30)
            update_step(run_id, "DataBricks", JobStatus.COMPLETED)
            update_step(run_id, "patternAnalyst", JobStatus.RUNNING)
            time.sleep(30)
            update_step(run_id, "patternAnalyst", JobStatus.COMPLETED)
            update_step(run_id, "fourCAnalyst", JobStatus.RUNNING)
            time.sleep(20)
            update_step(run_id, "fourCAnalyst", JobStatus.COMPLETED)
            update_step(run_id, "marketingStrategist", JobStatus.RUNNING)
            time.sleep(20)
            update_step(run_id, "marketingStrategist", JobStatus.COMPLETED)
            update_step(run_id, "videoMakingAgent", JobStatus.RUNNING)
            time.sleep(10)
            update_step(run_id, "videoMakingAgent", JobStatus.RUNNING)
            return update_run(run_id, {
            "status": JobStatus.COMPLETED.value,
            "analysisRef": analysis_id,
            })
        else:
            update_run(run_id, {"status": JobStatus.RUNNING.value})

            """
            # ---- Databricks ----
            brick_id = ingest_azure_adls_from_url(req).get("run_id")
            while True:
                brick_status = get_databricks_run_status(brick_id)
                print("Databricks job status:", brick_status)
                if (brick_status.get("life_cycle_state") == "TERMINATED"):
                    break

                time.sleep(10)
            """
            
            update_step(run_id, "DataBricks", JobStatus.COMPLETED)
            # ---- Pattern Analyst ----
            update_step(run_id, "patternAnalyst", JobStatus.RUNNING)
            last = Pattern_Analyst.invoke(
                    {
                        "messages": [{
                            "role": "user",
                            "content": (
                                "Analyze the following customer cluster data and provide insights "
                                "according to your role as Pattern Analyst Agent. "
                                "Use the get_record() tool to inspect the DataFrame as needed. "
                                "DO NOT EXCEED 10 QUERIES."
                            )
                        }]
                    },
                    config={"recursion_limit": 25},
                )["messages"][-1]
            text = extract_json_block(last.content if isinstance(last.content, str) else last.content[0].get("text", ""))
            pattern_output = pattern_parser.parse(
                text
            )
            update_step(run_id, "patternAnalyst", JobStatus.COMPLETED)

            # ---- FourC Analyst ----
            update_step(run_id, "fourCAnalyst", JobStatus.RUNNING)
            last = FourC_analyst.invoke(
                    {
                        "messages": [{
                            "role": "user",
                            "content": (
                                "Based on the following Pattern Analysis of the customer cluster data, "
                                "provide a detailed FourC analysis in HTML format defining the problem "
                                "of why customers churn from the perspectives of Customer, Cost, "
                                "Convenience, and Communication using get_record() and search_web() "
                                "where relevant, and return only valid JSON with those four fields. "
                                f"\n\nCONTEXT OF THIS CLUSTER\n---\n{pattern_output.model_dump_json(indent=2)}\n---\n "
                                "DO NOT EXCEED 10 ITERATIONS."
                            )
                        }]
                    },
                    config={"recursion_limit": 25},
                )["messages"][-1]
            text = extract_json_block(last.content if isinstance(last.content, str) else last.content[0].get("text", ""))
            FourC_output = fourc_parser.parse(
                text
            )
            update_step(run_id, "fourCAnalyst", JobStatus.COMPLETED)

            # ---- Marketing Strategist ----
            update_step(run_id, "marketingStrategist", JobStatus.RUNNING)
            last = Marketing_strategist.invoke(
                    {
                        "messages": [{
                            "role": "user",
                            "content": (
                                "Based on the following FourC analysis of the customer cluster data, "
                                "provide a comprehensive marketing strategy including Strategy Summary, "
                                "Strategic Pillars, Tactical Execution, Budget Allocation, Financial Projection, "
                                "and Expected Impact in HTML format using search_web() where relevant, "
                                "and return only valid JSON with those fields. "
                                f"\n\nCONTEXT OF THIS CLUSTER\n---\n{pattern_output.model_dump_json(indent=2)}\n---\n"
                                "\n---\n4C ANALYSIS\n---\n"
                                f"{FourC_output.model_dump_json(indent=2)} DO NOT EXCEED 15 QUERIES."
                            )
                        }]
                    },
                    config={"recursion_limit": 25},
                )["messages"][-1]
            print(last)
            text = extract_json_block(last.content if isinstance(last.content, str) else last.content[0].get("text", ""))
            
            Marketing_strategist_output = strategist_parser.parse(
                text
            )

            update_step(run_id, "marketingStrategist", JobStatus.COMPLETED)

            video_link = f"output/{Marketing_strategist_output.Title}.mp4"
            video_url = video_making_agent(prompt.Video_Making_Agent_Config.prompt_template, Marketing_strategist_output.model_dump_json(indent=2), video_making_agent_parser, file_name=video_link)

            # ---- Save final outputs into analyses/{analysisId} ----
            cluster_obj = Clusters(
                pattern_output=pattern_output,
                fourc_output=FourC_output,
                marketing_strategist_output=Marketing_strategist_output,
                number_of_customers=100,       # example
                definition="High-risk one-time buyers",
                video_link=[video_url + sas_token],  # optional
            )

            analysis_id = save_analysis_to_firestore(
                cluster_number=3,
                cluster=cluster_obj,
                total_loss=500.0,
            )

            update_run(run_id, {
                "status": JobStatus.COMPLETED.value,
                "analysisRef": analysis_id,
            })
    except Exception as e: 
        print(f"❌ Job failed: {e}")
        update_run(run_id, {
            "status": JobStatus.FAILED.value,
            "error": str(e),
        })



@app.post("/analyze_cluster")
def analyze_cluster(req: AzureUrlIngestRequest2, background_tasks: BackgroundTasks):
    run_id = create_run_doc()

    # run job in background
    background_tasks.add_task(run_analyze_cluster_job, run_id, req)

    return {
        "run_id": run_id,
        "status": JobStatus.QUEUED,
        "message": "Cluster analysis started.",
    }

@app.post("/check_run")
def check_run(query: RunCheckRequest):
    run_id = query.run_id
    doc = db.collection(RUNS_COLLECTION).document(run_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Run ID not found")
    doc_data = doc.to_dict()
    steps = doc_data.get("steps", {})
    progress = []
    for steps_name, step_info in steps.items():
        progress.append(step_info.get("status", "queued"))
    return {"progress": sum([1 for s in progress if s == "completed"])}

@app.get("/implementation")
def implementation_planner_agent(req: ImplementationInput):
    if solution_idx == 0:
        result = db.collection(IMPLEMENTATION_COLLECTION).document("XYllWNGqJbZitKvG5ROI").get()
    elif solution_idx == 1:
        result = db.collection(IMPLEMENTATION_COLLECTION).document("KYfY5tlLImNXvgTUSHJR").get()
    else:
        analysis_id = req.analysis_id
        solution_idx = req.solution_idx

        analysis_doc = db.collection(ANALYSES_COLLECTION).document(analysis_id).get()
        if not analysis_doc.exists:
            raise HTTPException(status_code=404, detail="Analysis ID not found")
        
        analysis_data = analysis_doc.to_dict()

        # Extract relevant context for the specified solution index
        try:
            strategy_info = analysis_data["clusters"][0]['strategy']
        except IndexError:
            raise HTTPException(status_code=400, detail="Invalid solution index")
        full_prompt = prompt.Implementation_Agent_Config.prompt_template + f"\n\nCONTEXT:\n{strategy_info}\n\n"

        llm_resp = llm.invoke(full_prompt)
        raw_text = llm_resp.content[0]['text'] if hasattr(llm_resp, "content") else str(llm_resp)
        parser = PydanticOutputParser(pydantic_object=prompt.Implementation_Agent_Config.format)
        result = parser.parse(extract_json_block(raw_text))
        print("📜 Implementation Plan Result:", result)

        analysis_doc = db.collection(IMPLEMENTATION_COLLECTION).document().set(
            result.model_dump()
        )

    return {"implementation_plan": result}

@app.get("/analysis_result")
def analysis_result():

    analysis_doc = db.collection(ANALYSES_COLLECTION).document("hNXinSgZn1YBr9IkCJVF").get()
    if not analysis_doc.exists:
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    
    analysis_data = analysis_doc.to_dict()

    return {"analysis_result": analysis_data}

#@app.get("/workplace-conversation")
#def workplace_conversation(selected_tasks: str):

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
