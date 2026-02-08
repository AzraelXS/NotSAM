#!/usr/bin/env python3
"""
Fix Elasticsearch datastream mapping to support tool result archiving

This script will:
1. Delete the existing datastream
2. Delete the index template
3. Recreate both with the updated mapping

WARNING: This will delete all data in the datastream!
Make sure you have a backup or are okay with losing the data.
"""

import requests
import json
from config import load_config

def fix_elasticsearch_mapping():
    """Fix the datastream mapping by recreating it"""
    
    # Load config
    config = load_config()
    es_config = config.elasticsearch
    
    if not es_config.enabled:
        print("❌ Elasticsearch is not enabled in config.json")
        return
    
    host = es_config.host
    datastream_name = es_config.datastream_name
    template_name = f"{datastream_name}-template"
    
    # Auth
    auth = None
    if es_config.username and es_config.password:
        auth = (es_config.username, es_config.password)
    
    print(f"🔧 Fixing Elasticsearch mapping for: {datastream_name}")
    print(f"   Host: {host}")
    
    # Ask for confirmation
    response = input("\n⚠️  This will DELETE all data in the datastream. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Aborted")
        return
    
    print("\n1️⃣ Deleting datastream...")
    response = requests.delete(
        f"{host}/_data_stream/{datastream_name}",
        auth=auth,
        verify=es_config.verify_ssl
    )
    if response.status_code in [200, 404]:
        print(f"   ✅ Datastream deleted (or didn't exist)")
    else:
        print(f"   ⚠️  Status: {response.status_code}")
    
    print("\n2️⃣ Deleting index template...")
    response = requests.delete(
        f"{host}/_index_template/{template_name}",
        auth=auth,
        verify=es_config.verify_ssl
    )
    if response.status_code in [200, 404]:
        print(f"   ✅ Template deleted (or didn't exist)")
    else:
        print(f"   ⚠️  Status: {response.status_code}")
    
    print("\n3️⃣ Creating new template with updated mapping...")
    template = {
        "index_patterns": [f"{datastream_name}*"],
        "data_stream": {},
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "memory_type": {"type": "keyword"},
                    "role": {"type": "keyword"},
                    "tool_name": {"type": "keyword"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    },
                    # Tool result archive fields (NEW!)
                    "message_index": {"type": "integer"},
                    "tool_args": {"type": "object", "enabled": True},
                    "tool_result": {"type": "text"},
                    # Plan-specific fields
                    "plan_id": {"type": "keyword"},
                    "description": {"type": "text"},
                    "status": {"type": "keyword"},
                    "steps": {"type": "object", "enabled": True},
                    "total_steps": {"type": "integer"},
                    "completed_steps": {"type": "integer"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "metadata": {"type": "object", "enabled": True}
                }
            }
        }
    }
    
    response = requests.put(
        f"{host}/_index_template/{template_name}",
        json=template,
        auth=auth,
        verify=es_config.verify_ssl
    )
    if response.status_code in [200, 201]:
        print(f"   ✅ Template created")
    else:
        print(f"   ❌ Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    print("\n4️⃣ Creating datastream...")
    response = requests.put(
        f"{host}/_data_stream/{datastream_name}",
        auth=auth,
        verify=es_config.verify_ssl
    )
    if response.status_code in [200, 201]:
        print(f"   ✅ Datastream created")
    else:
        print(f"   ❌ Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    print("\n✅ Elasticsearch mapping fixed!")
    print("   SAM can now archive tool results without errors.")

if __name__ == "__main__":
    fix_elasticsearch_mapping()
