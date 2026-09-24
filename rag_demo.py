

"""LangChain 单文档问答 Demo：文档加载 → 分割 → Milvus 向量化 → 检索问答。"""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

# ── 配置常量 ──────────────────────────────────────────────
DOC_PATH = os.path.join(os.path.dirname(__file__), "sample_doc.txt")
MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_demo.db")  # 默认 Milvus Lite 文件模式
COLLECTION_NAME = "langchain_demo"

# Embedding 配置（复用 SiliconFlow 的 bge 模型）
EMBEDDING_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
EMBEDDING_API_KEY = os.getenv("LLM_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")


def _build_llm():
    """构建 LangChain 的 ChatOpenAI 实例（兼容 OpenAI 协议）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=EMBEDDING_BASE_URL,
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        streaming=True,
    )


def _build_embeddings():
    """构建 Embedding 实例（使用 SiliconFlow 的 bge 模型，通过自定义封装绕过兼容性问题）。"""
    from langchain_core.embeddings import Embeddings

    class SiliconFlowEmbeddings(Embeddings):
        """兼容 SiliconFlow 的 Embedding 封装。"""
        def __init__(self, base_url: str, api_key: str, model: str):
            self.base_url = base_url
            self.api_key = api_key
            self.model = model
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            })

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            resp = self._session.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": texts},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # 按原始索引排序，保证顺序一致
            results = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in results]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    return SiliconFlowEmbeddings(EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL)


def load_and_split_documents(path: str):
    """加载文本文件并按段落分割。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with open(path, encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " "],
    )
    chunks = splitter.create_documents([text])
    print(f"📄 文档已加载，共分割为 {len(chunks)} 个片段。")
    return chunks


def build_vector_store(chunks, embeddings):
    """将文档片段向量化并存入 Milvus。"""
    from langchain_milvus import Milvus

    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        drop_old=True,  # 每次运行重建集合
    )
    print(f"✅ 向量已存入 Milvus（URI: {MILVUS_URI}）。")
    return vectorstore


def build_qa_chain(vectorstore, llm):
    """使用 LCEL 构建检索问答链。"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "请根据以下参考文档回答问题。如果文档中没有相关信息，请如实说明。\n\n"
        "参考文档：\n{context}\n\n"
        "问题：{question}\n\n"
        "回答："
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def main() -> None:
    print("=" * 50)
    print("   📚 LangChain 单文档问答 Demo（Milvus）")
    print("=" * 50)

    # 1. 加载 & 分割文档
    if not os.path.exists(DOC_PATH):
        print(f"❌ 未找到示例文档: {DOC_PATH}")
        sys.exit(1)
    chunks = load_and_split_documents(DOC_PATH)

    # 2. 构建 Embedding 并存入 Milvus
    embeddings = _build_embeddings()
    vectorstore = build_vector_store(chunks, embeddings)

    # 3. 构建问答链
    llm = _build_llm()
    qa, retriever = build_qa_chain(vectorstore, llm)

    # 4. 交互式问答
    print("\n💡 输入问题开始问答，输入 /exit 退出。\n")
    while True:
        try:
            question = input("🧑 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if not question:
            continue
        if question.lower() in ("/exit", "/quit"):
            print("👋 再见！")
            break

        try:
            answer = qa.invoke(question)
            print(f"\n🤖 回答: {answer}\n")

            # 显示引用的源文档片段
            docs = retriever.invoke(question)
            if docs:
                print(f"📎 参考了 {len(docs)} 个文档片段:")
                for i, doc in enumerate(docs, 1):
                    preview = doc.page_content[:80].replace("\n", " ")
                    print(f"   [{i}] {preview}...")
                print()
        except Exception as e:
            print(f"\n⚠️  出错了: {e}\n")


if __name__ == "__main__":
    main()
