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
    model="gemini-2.5-flash",
)
