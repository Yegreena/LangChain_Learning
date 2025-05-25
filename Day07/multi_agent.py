# agent_main.py

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub

# 导入工具
from tools import image_tool, json_tool, wiki_tool, weather

# 模型初始化
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 工具列表
tools = [image_tool, json_tool, wiki_tool, weather]

# 使用官方函数调用 Agent 的 prompt 模板
prompt = hub.pull("hwchase17/openai-functions-agent")

# 创建 agent 和执行器
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    # 你可以修改测试内容
    query = "请查询北京的天气。"
    response = agent_executor.invoke({"input": query})
    print("\n=== Agent Response ===")
    print(response["output"])
