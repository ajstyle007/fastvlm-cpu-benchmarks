import os
import pandas as pd
from io import BytesIO
from PIL import Image

# Setup absolute paths to your cached files
CACHE_DIR = "./gqa_cache"
TEXT_FILE = os.path.join(CACHE_DIR, "val-00000-of-00001.parquet")
IMAGE_FILES = [
    os.path.join(CACHE_DIR, "val-00000-of-00003.parquet"),
    os.path.join(CACHE_DIR, "val-00001-of-00003.parquet"),
    os.path.join(CACHE_DIR, "val-00002-of-00003.parquet")
]

def explore_text_parquet():
    print("=" * 60)
    print(f"📊 ANALYZING TEXT FILE: {os.path.basename(TEXT_FILE)}")
    print("=" * 60)
    
    # Read metadata first to prevent memory overhead
    df_text = pd.read_parquet(TEXT_FILE)
    
    print(f"🔹 Total rows (Questions) found: {len(df_text)}")
    print("\n🔹 Available Columns & Data Types:")
    print(df_text.dtypes)
    
    # Analyze target answers
    print("\n🔹 Top 10 Most Common Answers in Dataset:")
    print(df_text['answer'].value_counts().head(10))
    
    # Check binary vs non-binary ratio
    yes_count = df_text['answer'].str.lower().str.strip().eq('yes').sum()
    no_count = df_text['answer'].str.lower().str.strip().eq('no').sum()
    total_binary = yes_count + no_count
    
    print(f"\n🔹 Binary Filter Metrics:")
    print(f"   - 'Yes' count: {yes_count}")
    print(f"   - 'No' count: {no_count}")
    print(f"   - Total Binary Samples: {total_binary} ({round((total_binary/len(df_text))*100, 2)}% of dataset)")
    
    print("\n📝 PREVIEWING FIRST 3 SAMPLES:")
    for idx, row in df_text.head(3).iterrows():
        print(f"\n--- Sample {idx+1} (ID: {row.get('id') or row.get('question_id')}) ---")
        print(f"🖼️ Image Reference ID: {row.get('imageId')}")
        print(f"❓ Question: {row['question']}")
        print(f"🎯 Full Answer: {row.get('fullAnswer')} (Short: {row['answer']})")
        if 'types' in row:
            print(f"⚙️ Structural Type: {row['types']}")
            
    return df_text

import pyarrow.parquet as pq

def explore_image_parquet():
    print("\n" + "=" * 60)
    print(f"🖼️ ANALYZING IMAGE FILE SHARDS")
    print("=" * 60)
    
    for i, img_path in enumerate(IMAGE_FILES):
        if not os.path.exists(img_path):
            print(f"❌ Cannot find shard: {img_path}")
            continue
            
        try:
            # Open the file structure directly without loading rows
            parquet_file = pq.ParquetFile(img_path)
            
            # Print column schema layout
            print(f"🔹 Shard {i} ({os.path.basename(img_path)}):")
            print(f"   - Column Names in Schema: {parquet_file.schema.names}")
            print(f"   - Total rows in this shard file: {parquet_file.metadata.num_rows}")
            
            # Read exactly the first row group safely
            first_row_batch = next(parquet_file.iter_batches(batch_size=1, columns=['id', 'image']))
            df_img_head = first_row_batch.to_pandas()
            
            # Pull the data format of the image column
            first_row_img = df_img_head['image'].iloc[0]
            print(f"   - Type of 'image' field inside parquet row: {type(first_row_img)}")
            
            if isinstance(first_row_img, dict):
                print(f"   - Dictionary keys inside image row feature: {list(first_row_img.keys())}")
                if 'bytes' in first_row_img:
                    img_bytes = first_row_img['bytes']
                    with Image.open(BytesIO(img_bytes)) as img:
                        print(f"   - Decoded PIL verification size: {img.size} | Mode: {img.mode}")
            elif hasattr(first_row_img, 'get'):
                print(f"   - Accessible dict keys: {list(first_row_img.keys())}")
            else:
                print(f"   - Raw Sample Data Content (Truncated): {str(first_row_img)[:100]}")
                
        except Exception as e:
            print(f"   - Error inspecting this shard layout: {e}")
            
        print("-" * 50)

if __name__ == "__main__":
    if not os.path.exists(TEXT_FILE):
        print(f"❌ Error: Could not find text file at {TEXT_FILE}. Run the main script first to download.")
    else:
        explore_text_parquet()
        explore_image_parquet()