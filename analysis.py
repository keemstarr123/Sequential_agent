import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.tools import tool
from prompt import Pattern_Analyst_Prompt
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import prompt
from googleapiclient.discovery import build
import pprint
import asyncio
import httpx
from bs4 import BeautifulSoup
import time
from google import genai
import cv2
from http import HTTPStatus
from dashscope import VideoSynthesis
import dashscope
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

# Very simple in-memory job store
jobs: Dict[str, Dict[str, Any]] = {}

tool_call_state = {"count": 0, "limit": 10}


load_dotenv()
gcp_key = os.getenv("GOOGLE_API_KEY")
dashscope_key = os.getenv("DASHSCOPE_API_KEY")
cse_id = os.getenv("GOOGLE_CSE_ID")
client = genai.Client(api_key=gcp_key)
openai_client =  OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'


class Cluster:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.num = get_num(self.df)
        


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
        return {"error": "Recursion limit exceeded. Summarize your findings and stop."}
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
            return {"error": "Recursion limit exceeded. Summarize your findings and stop."}
        try:
            return eval(pd_statement) 
        except RecursionError:
            return "⚠️ RecursionError detected. Halting to prevent infinite loop."   
        except Exception as e:
            return str(e)
        


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

def video_making_agent(template: str, context: str, parser: PydanticOutputParser):
    full_prompt = template + f"\n\nCONTEXT:\n{context}\n\n"

    llm_resp = llm.invoke(full_prompt)
    raw_text = llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)
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

    print("🎬 Final combined video: combined_output.mp4")
    return result


#def implementation_planner(template, context, parser: PydanticOutputParser):


           

pattern_parser = PydanticOutputParser(pydantic_object=prompt.Pattern_Analyst_Config.format)
fourc_parser = PydanticOutputParser(pydantic_object=prompt.FourC_Analyst_Config.format)
strategist_parser = PydanticOutputParser(pydantic_object=prompt.Marketing_Strategist_Config.format) 
video_making_agent_parser = PydanticOutputParser(pydantic_object=prompt.Video_Making_Agent_Config.format)
claude_api_key = os.getenv("CLAUDE_API")
cluster = Cluster("./data/cluster_3.csv")
llm = ChatGoogleGenerativeAI(google_api_key=gcp_key, model="gemini-2.5-pro")



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

def run_analyze_cluster_job(job_id: str):
    try:
        jobs[job_id]["status"] = JobStatus.RUNNING

        pattern_output = pattern_parser.parse(Pattern_Analyst.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    "Analyze the following customer cluster data and provide insights "
                    "according to your role as Pattern Analyst Agent. "
                    "Use the get_record() tool to inspect the DataFrame as needed. "
                    "DO NOT EXCEED 10 QUERIES."
                )
            }]
        }, config={'recursion_limit': 25})["messages"][-1].content[0]['text'])

        print(f"Pattern done, proceeding to 4C analysis...\n Insights:\n{pattern_output.model_dump_json(indent=2)}")

        FourC_output = fourc_parser.parse(FourC_analyst.invoke({
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
        }, config={'recursion_limit': 25})["messages"][-1].content[0]['text'])

        print(f"4C analysis done, proceeding to marketing strategy formulation... \n Insights:\n{FourC_output.model_dump_json(indent=2)}")

        Marketing_strategist_output = strategist_parser.parse(Marketing_strategist.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    "Based on the following FourC analysis of the customer cluster data, "
                    "provide a comprehensive marketing strategy including Strategy Summary, "
                    "Strategic Pillars, Tactical Execution, Budget Allocation, Financial Projection, "
                    "and Expected Impact in HTML format using search_web() where relevant, "
                    "and return only valid JSON with those fields. "
                    f"\n\nCONTEXT OF THIS CLUSTER\n---\n{pattern_output.model_dump_json(indent=2)}\n---\n"
                    "\n---\n4C ANALAYIS\n---\n"
                    f"{FourC_output.model_dump_json(indent=2)} DO NOT EXCEED 15 QUERIES."
                )
            }]
        }, config={'recursion_limit': 25})["messages"][-1].content[0]['text'])

        print(f"Marketing strategy is completed, moving on towards video generation. \n Insights:\n{Marketing_strategist_output.model_dump_json(indent=2)}")

        # If you want to trigger video generation inside the job:
        # video_making_agent(...)

        # concat_videos_moviepy(['Video_1.mp4', 'Video_2.mp4'], out_path="combined_output.mp4", target_fps=30)

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["result"] = {
            "pattern_analysis": pattern_output.model_dump(),
            "fourc_analysis": FourC_output.model_dump(),
            "marketing_strategy": Marketing_strategist_output.model_dump(),
        }

    except Exception as e:
        print("❌ Job failed:", e)
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)

@app.post("/analyze_cluster")
def analyze_cluster():
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": JobStatus.QUEUED,
        "result": None,
        "error": None,
        "created_at": time.time(),
    }

    # Run the heavy job in the background (non-blocking)
    background_tasks.add_task(run_analyze_cluster_job, job_id)

    return {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "message": "Cluster analysis started.",
    }

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
