from dotenv import load_dotenv
import json
import os
import tempfile

os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

from langchain_google_vertexai import ChatVertexAI


load_dotenv()

gcp_secret = os.environ.get("GCP_CREDS_JSON")

if gcp_secret:
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(gcp_secret)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path

        creds_dict = json.loads(gcp_secret)
        os.environ["GOOGLE_CLOUD_PROJECT"] = creds_dict.get("project_id", "")

        print("✅ Secure Mode: โหลด Vertex AI Credentials สำเร็จ!")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
else:
    print("⚠️ Local Mode: ใช้ Credentials จากไฟล์ .env")


intent_model = ChatVertexAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
)

chat_model = ChatVertexAI(
    # Flash-Lite is sufficient for this constrained, retrieval-grounded health
    # response.  The deterministic guardrails and citation pipeline still run
    # unchanged, while this materially lowers generation latency.
    model=os.getenv("HEALTH_CHAT_MODEL", "gemini-2.5-flash-lite"),
    temperature=0,
    # 300 can truncate a Thai multi-result health summary mid-sentence. The
    # prompt still requests concise answers; this is only a safe completion
    # ceiling for cases containing several lab domains.
    max_output_tokens=512,
)
