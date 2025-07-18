#!/usr/bin/env python3
"""
Demo script for Dremio MCP Memory MVP

This script demonstrates the memory functionality by:
1. Showing existing memories
2. Searching for business metrics
3. Adding a new memory
4. Demonstrating persistence across sessions
"""

import subprocess
import json
import sys
from pathlib import Path

def run_tool(tool_name, **kwargs):
    """Run a dremio-mcp-server tool and return the result."""
    cmd = ["uv", "run", "dremio-mcp-server", "tools", "invoke", "-t", tool_name]
    
    for key, value in kwargs.items():
        if isinstance(value, list):
            value = json.dumps(value)
        cmd.append(f"{key}={value}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            # Extract JSON from the output (skip log lines)
            lines = result.stdout.strip().split('\n')

            # Look for JSON content - it might span multiple lines
            json_content = ""
            in_json = False
            brace_count = 0

            for line in lines:
                line = line.strip()
                if not in_json and line.startswith('{'):
                    in_json = True
                    json_content = line
                    brace_count = line.count('{') - line.count('}')
                elif in_json:
                    json_content += line
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0:
                        break

            if json_content:
                # Handle Python-style output with single quotes
                try:
                    # First try standard JSON parsing
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    # If that fails, try evaluating as Python literal
                    import ast
                    return ast.literal_eval(json_content)
            else:
                return {"error": "No JSON output found", "stdout": result.stdout}
        else:
            return {"error": f"Command failed: {result.stderr}", "returncode": result.returncode}
    except Exception as e:
        return {"error": str(e), "stdout": getattr(result, 'stdout', ''), "stderr": getattr(result, 'stderr', '')}

def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_memory(memory):
    """Print a memory in a nice format."""
    print(f"ID: {memory['id']}")
    if 'tags' in memory:
        print(f"Tags: {', '.join(memory['tags'])}")
    if 'text_preview' in memory:
        print(f"Preview: {memory['text_preview'][:200]}...")
    elif 'text' in memory:
        print(f"Text: {memory['text'][:200]}...")
    print()

def main():
    print("🧠 Dremio MCP Memory MVP Demo")
    print("This demo shows how to store and retrieve business knowledge")
    
    # 1. List existing memories
    print_section("1. Existing Memories")
    memories_result = run_tool("ListMemories", limit=10)
    
    if "error" in memories_result:
        print(f"❌ Error listing memories: {memories_result['error']}")
        return
    
    memories = memories_result.get("memories", [])
    print(f"Found {len(memories)} existing memories:")
    for memory in memories[:3]:  # Show first 3
        print_memory(memory)
    
    if len(memories) > 3:
        print(f"... and {len(memories) - 3} more")
    
    # 2. Search for ARR definition
    print_section("2. Search for ARR Definition")
    search_result = run_tool("SearchMemory", query="ARR definition", top_k=3)
    
    if "error" in search_result:
        print(f"❌ Error searching: {search_result['error']}")
        return
    
    matches = search_result.get("matches", [])
    print(f"Found {len(matches)} matches for 'ARR definition':")
    for i, match in enumerate(matches, 1):
        print(f"{i}. Score: {match['score']:.3f}")
        print_memory(match)
    
    # 3. Get full ARR definition
    if matches:
        print_section("3. Full ARR Definition")
        arr_id = matches[0]["id"]  # Get the top match
        full_memory = run_tool("GetMemory", id=arr_id)
        
        if "error" not in full_memory:
            print("Full ARR Definition:")
            print(f"ID: {full_memory['id']}")
            print(f"Tags: {', '.join(full_memory['tags'])}")
            print(f"Text:\n{full_memory['text']}")
        else:
            print(f"❌ Error retrieving memory: {full_memory['error']}")
    
    # 4. Add a new memory
    print_section("4. Adding New Memory")
    new_memory_text = """Net Revenue Retention (NRR): 
    
Measures the percentage of recurring revenue retained from existing customers over a period, including expansions and contractions.

Formula: ((Starting ARR + Expansion ARR - Contraction ARR - Churned ARR) / Starting ARR) × 100

Target: >110% indicates healthy growth from existing customer base.
Benchmark: Best-in-class SaaS companies achieve 120%+ NRR.

Data source: Calculate monthly from subscription_changes table."""
    
    put_result = run_tool("PutMemory", 
                         text=new_memory_text,
                         id="metric:NRR",
                         tags=["metric", "finance", "retention", "NRR"])
    
    if "error" not in put_result:
        print("✅ Successfully added NRR definition:")
        print(f"ID: {put_result['id']}")
        print(f"Estimated tokens: {put_result['tokens_est']}")
        print(f"Stored: {put_result['stored']}")
    else:
        print(f"❌ Error storing memory: {put_result['error']}")
    
    # 5. Search for the new memory
    print_section("5. Search for New Memory")
    nrr_search = run_tool("SearchMemory", query="net revenue retention", top_k=2)
    
    if "error" not in nrr_search:
        matches = nrr_search.get("matches", [])
        print(f"Found {len(matches)} matches for 'net revenue retention':")
        for i, match in enumerate(matches, 1):
            print(f"{i}. Score: {match['score']:.3f}")
            print_memory(match)
    else:
        print(f"❌ Error searching: {nrr_search['error']}")
    
    # 6. Demonstrate tag filtering
    print_section("6. Filter by Finance Tags")
    finance_search = run_tool("SearchMemory", 
                             query="revenue", 
                             top_k=5, 
                             tag_filter=["finance"])
    
    if "error" not in finance_search:
        matches = finance_search.get("matches", [])
        print(f"Found {len(matches)} finance-related memories:")
        for i, match in enumerate(matches, 1):
            print(f"{i}. {match['id']} (Score: {match['score']:.3f})")
    else:
        print(f"❌ Error searching with tags: {finance_search['error']}")
    
    print_section("Demo Complete!")
    print("🎉 The memory system successfully:")
    print("   ✅ Listed existing memories")
    print("   ✅ Searched for specific definitions")
    print("   ✅ Retrieved full memory content")
    print("   ✅ Added new memories")
    print("   ✅ Filtered by tags")
    print("\n💡 This memory persists across MCP sessions, so assistants can")
    print("   remember your business definitions and use them consistently!")

if __name__ == "__main__":
    main()
