# supabase_pull.py

import os
import pandas as pd

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("project-url")
SUPABASE_KEY = os.getenv("supabase-api-key")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials missing in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all(table_name="Procure_data", batch_size=1000):

    start = 0
    all_data = []

    while True:

        response = (
            supabase.table(table_name)
            .select("*")
            .range(start, start + batch_size - 1)
            .execute()
        )

        batch = response.data or []
        all_data.extend(batch)

        if len(batch) < batch_size:
            break

        start += batch_size

    return pd.DataFrame(all_data)


if __name__ == "__main__":

    df = fetch_all()

    print("Total rows:", len(df))
    print(df.head())