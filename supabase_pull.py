import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


SUPABASE_URL =  os.getenv("project-url")
SUPABASE_KEY = os.getenv("supabase-api-key")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_all():
    batch_size = 1000
    start = 0
    all_data = []

    while True:
        rep = (
            supabase.table("Procure_data")
            .select("*")
            .range(start, start + batch_size - 1)
            .execute()
        )

        batch = rep.data
        all_data.extend(batch)

        if len(batch) < batch_size:
            break

        start += batch_size

    return pd.DataFrame(all_data)

df = fetch_all()

print("Total rows:", len(df))
print(df.head())