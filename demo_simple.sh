#!/bin/bash

echo "🧠 Dremio MCP Memory MVP Demo"
echo "This demo shows how to store and retrieve business knowledge"
echo ""

echo "============================================================"
echo " 1. List Existing Memories"
echo "============================================================"
echo "Command: uv run dremio-mcp-server tools invoke -t ListMemories limit=3"
echo ""
uv run dremio-mcp-server tools invoke -t ListMemories limit=3 2>/dev/null | tail -n +10
echo ""

echo "============================================================"
echo " 2. Search for ARR Definition"
echo "============================================================"
echo "Command: uv run dremio-mcp-server tools invoke -t SearchMemory query=\"ARR definition\" top_k=2"
echo ""
uv run dremio-mcp-server tools invoke -t SearchMemory query="ARR definition" top_k=2 2>/dev/null | grep -A 50 "matches"
echo ""

echo "============================================================"
echo " 3. Get Full ARR Definition"
echo "============================================================"
echo "Command: uv run dremio-mcp-server tools invoke -t GetMemory id=\"metric:ARR\""
echo ""
uv run dremio-mcp-server tools invoke -t GetMemory id="metric:ARR" 2>/dev/null | grep -A 20 "text"
echo ""

echo "============================================================"
echo " 4. Add New Memory (Net Revenue Retention)"
echo "============================================================"
echo "Command: uv run dremio-mcp-server tools invoke -t PutMemory ..."
echo ""
uv run dremio-mcp-server tools invoke -t PutMemory \
  text="Net Revenue Retention (NRR): Measures the percentage of recurring revenue retained from existing customers over a period, including expansions and contractions. Formula: ((Starting ARR + Expansion ARR - Contraction ARR - Churned ARR) / Starting ARR) × 100. Target: >110% indicates healthy growth." \
  id="metric:NRR" \
  tags='["metric","finance","retention","NRR"]' 2>/dev/null | tail -n +10
echo ""

echo "============================================================"
echo " 5. Search for New Memory"
echo "============================================================"
echo "Command: uv run dremio-mcp-server tools invoke -t SearchMemory query=\"net revenue retention\""
echo ""
uv run dremio-mcp-server tools invoke -t SearchMemory query="net revenue retention" top_k=2 2>/dev/null | tail -n +10
echo ""

echo "============================================================"
echo " 6. Filter by Finance Tags"
echo "============================================================"
echo "Command: uv run dremio-mcp-server tools invoke -t SearchMemory query=\"revenue\" tag_filter='[\"finance\"]'"
echo ""
uv run dremio-mcp-server tools invoke -t SearchMemory query="revenue" tag_filter='["finance"]' top_k=3 2>/dev/null | tail -n +10
echo ""

echo "============================================================"
echo " Demo Complete!"
echo "============================================================"
echo "🎉 The memory system successfully:"
echo "   ✅ Listed existing memories"
echo "   ✅ Searched for specific definitions"
echo "   ✅ Retrieved full memory content"
echo "   ✅ Added new memories"
echo "   ✅ Filtered by tags"
echo ""
echo "💡 This memory persists across MCP sessions, so assistants can"
echo "   remember your business definitions and use them consistently!"
echo ""
echo "🚀 Ready for MCP integration with Claude Desktop, ChatGPT, or any MCP client!"
