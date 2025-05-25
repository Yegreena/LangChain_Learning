# tools.py

import json
import asyncio
import aiohttp
from langchain.agents import Tool

# 图片理解工具（模拟）
def image_description_tool(image_path: str) -> str:
    return f"模拟图片描述：图片路径是 {image_path}，可能包含动物或风景。"

image_tool = Tool.from_function(
    func=image_description_tool,
    name="image_description",
    description="输入图片路径，返回图片内容简要描述"
)

# JSON结构化输出工具
def structured_output_tool(text: str) -> str:
    data = {
        "原始内容": text,
        "字数": len(text),
        "关键词": text.split()[:3]
    }
    return json.dumps(data, ensure_ascii=False, indent=2)

json_tool = Tool.from_function(
    func=structured_output_tool,
    name="json_formatter",
    description="将输入文本结构化为 JSON 格式，包含原文、字数、关键词"
)

# 异步维基百科搜索（用aiohttp）
async def wiki_search_async(keyword: str) -> str:
    url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{keyword}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("extract", "未找到相关信息。")
            else:
                return "请求失败，可能没有该词条。"

def wiki_search(keyword: str) -> str:
    return asyncio.run(wiki_search_async(keyword))

wiki_tool = Tool.from_function(
    func=wiki_search,
    name="wiki_search",
    description="输入关键词，异步调用维基百科API返回简介"
)

# other_tools.py

from langchain_community.tools.tavily_search import TavilySearchResults

# 天气查询模拟
def weather_tool(city: str) -> str:
    return f"{city} 当前天气：晴，25°C，微风。"

weather = Tool.from_function(
    func=weather_tool,
    name="weather",
    description="输入城市名，返回模拟天气"
)

# Tavily网页搜索
# search_tool = TavilySearchResults(max_results=1)
