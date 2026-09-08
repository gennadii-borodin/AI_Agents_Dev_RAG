import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI


load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"


# curated analyst notes. mentions[] link a note to KG entities by name.
# intentionally: no note says who competes with whom — that lives only in the KG.
DOCUMENTS = [
    {
        "title": "Mistral bets on open weights",
        "text": (
            "Mistral AI continues to release open-weight models such as Mixtral. "
            "The company argues that openness speeds adoption among European enterprises, "
            "but says operational costs for serving large models remain high."
        ),
        "mentions": ["Mistral AI", "Mixtral"],
        "keywords": ["open-weights", "adoption", "costs"],
    },
    {
        "title": "OpenAI expands red-teaming",
        "text": (
            "OpenAI described a broader red-teaming program around GPT-4 deployments. "
            "External testers probe misuse scenarios before release, and findings feed "
            "into policy and monitoring updates."
        ),
        "mentions": ["OpenAI", "GPT-4"],
        "keywords": ["safety", "red-teaming", "monitoring"],
    },
    {
        "title": "Anthropic on constitutional AI",
        "text": (
            "Anthropic published notes on constitutional AI used with Claude 3. "
            "The approach encodes explicit principles into training and critique loops "
            "to reduce harmful responses without only relying on human preference labels."
        ),
        "mentions": ["Anthropic", "Claude 3"],
        "keywords": ["safety", "constitutional-ai", "principles"],
    },
    {
        "title": "EU AI Act obligations for GPAI",
        "text": (
            "The EU AI Act introduces transparency and risk-management duties for "
            "general-purpose AI providers. Documentation, copyright summaries, and "
            "incident reporting are recurring compliance themes."
        ),
        "mentions": ["EU AI Act"],
        "keywords": ["compliance", "transparency", "risk-management"],
    },
    {
        "title": "European labs face compliance load",
        "text": (
            "Analysts report that Mistral AI and Aleph Alpha are hiring policy staff "
            "to prepare technical documentation required under the EU AI Act. "
            "Smaller labs worry evaluation and reporting overhead will slow releases."
        ),
        "mentions": ["Mistral AI", "Aleph Alpha", "EU AI Act"],
        "keywords": ["compliance", "documentation", "policy"],
    },
    {
        "title": "DeepMind and the UK safety agenda",
        "text": (
            "DeepMind's Gemini work is often discussed alongside the UK's AI safety "
            "institutions. Researchers emphasize evaluation suites for dangerous "
            "capabilities rather than only consumer-facing refusals."
        ),
        "mentions": ["DeepMind", "Gemini"],
        "keywords": ["safety", "evaluation", "capabilities"],
    },
]


def erase_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def build_knowledge_graph(tx):
    # structured facts only — not copied from the documents above
    tx.run("""
        CREATE
            (usa:Country {name: 'USA', region: 'US'}),
            (france:Country {name: 'France', region: 'EU'}),
            (germany:Country {name: 'Germany', region: 'EU'}),
            (uk:Country {name: 'UK', region: 'UK'}),

            (mistral:Company {name: 'Mistral AI'}),
            (aleph:Company {name: 'Aleph Alpha'}),
            (openai:Company {name: 'OpenAI'}),
            (anthropic:Company {name: 'Anthropic'}),
            (deepmind:Company {name: 'DeepMind'}),

            (mixtral:Model {name: 'Mixtral', kind: 'foundation'}),
            (mistral_large:Model {name: 'Mistral Large', kind: 'foundation'}),
            (luminous:Model {name: 'Luminous', kind: 'foundation'}),
            (gpt4:Model {name: 'GPT-4', kind: 'foundation'}),
            (claude3:Model {name: 'Claude 3', kind: 'foundation'}),
            (gemini:Model {name: 'Gemini', kind: 'foundation'}),

            (ai_act:Regulation {name: 'EU AI Act'}),
            (us_eo:Regulation {name: 'US AI Executive Order'}),

            (mistral)-[:HEADQUARTERED_IN]->(france),
            (aleph)-[:HEADQUARTERED_IN]->(germany),
            (openai)-[:HEADQUARTERED_IN]->(usa),
            (anthropic)-[:HEADQUARTERED_IN]->(usa),
            (deepmind)-[:HEADQUARTERED_IN]->(uk),

            (mistral)-[:DEVELOPS]->(mixtral),
            (mistral)-[:DEVELOPS]->(mistral_large),
            (aleph)-[:DEVELOPS]->(luminous),
            (openai)-[:DEVELOPS]->(gpt4),
            (anthropic)-[:DEVELOPS]->(claude3),
            (deepmind)-[:DEVELOPS]->(gemini),

            (mistral)-[:COMPETES_WITH]->(openai),
            (mistral)-[:COMPETES_WITH]->(anthropic),
            (openai)-[:COMPETES_WITH]->(anthropic),

            (ai_act)-[:APPLIES_IN]->(france),
            (ai_act)-[:APPLIES_IN]->(germany),
            (us_eo)-[:APPLIES_IN]->(usa)
        """)


def build_lexical_graph(tx):
    # documents hang off the same entity nodes as the knowledge graph
    for doc in DOCUMENTS:
        tx.run(
            "CREATE (d:Document {title: $title, text: $text})",
            title=doc["title"],
            text=doc["text"],
        )
        for mention in doc["mentions"]:
            tx.run(
                """
                MATCH (d:Document {title: $title})
                MATCH (e)
                WHERE (e:Company OR e:Model OR e:Regulation) AND e.name = $mention
                MERGE (d)-[:MENTIONS]->(e)
                """,
                title=doc["title"],
                mention=mention,
            )
        for kw in doc["keywords"]:
            tx.run(
                """
                MATCH (d:Document {title: $title})
                MERGE (k:Keyword {name: $kw})
                MERGE (d)-[:HAS_KEYWORD]->(k)
                """,
                title=doc["title"],
                kw=kw,
            )


def ask_llm(question, context):
    prompt = (
        "Answer using only the retrieved context. "
        "If the context is not enough, say what is missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def format_context(knowledge_rows, document_rows):
    parts = ["[knowledge graph]"]
    for row in knowledge_rows:
        parts.append(str(dict(row)))

    parts.append("[documents]")
    for row in document_rows:
        parts.append(f"{row['title']}: {row['text']}")

    return "\n".join(parts)


def naive_keyword_retrieve(session, terms):
    # stand-in for classic RAG: no relationships, only text/keyword overlap
    return session.run(
        """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_KEYWORD]->(k:Keyword)
        WITH d, collect(k.name) AS keywords
        WHERE any(term IN $terms WHERE
            toLower(d.text) CONTAINS toLower(term)
            OR toLower(d.title) CONTAINS toLower(term)
            OR any(kw IN keywords WHERE toLower(kw) CONTAINS toLower(term))
        )
        RETURN d.title AS title, d.text AS text, keywords
        ORDER BY title
        """,
        terms=terms,
    ).data()


def retrieve_us_competitors_of_mistral(session):
    knowledge = session.run("""
        MATCH (mistral:Company {name: 'Mistral AI'})-[:COMPETES_WITH]-(comp:Company)
              -[:HEADQUARTERED_IN]->(c:Country {region: 'US'})
        OPTIONAL MATCH (comp)-[:DEVELOPS]->(model:Model)
        RETURN comp.name AS company, c.name AS country,
               collect(DISTINCT model.name) AS models
        ORDER BY company
    """).data()

    documents = session.run("""
        MATCH (mistral:Company {name: 'Mistral AI'})-[:COMPETES_WITH]-(comp:Company)
              -[:HEADQUARTERED_IN]->(:Country {region: 'US'})
        MATCH (d:Document)-[:MENTIONS]->(comp)
        OPTIONAL MATCH (d)-[:HAS_KEYWORD]->(k:Keyword)
        RETURN DISTINCT d.title AS title, d.text AS text,
               comp.name AS about_company,
               collect(DISTINCT k.name) AS keywords
        ORDER BY title
    """).data()

    return knowledge, documents


def retrieve_eu_foundation_compliance(session):
    knowledge = session.run("""
        MATCH (comp:Company)-[:HEADQUARTERED_IN]->(country:Country {region: 'EU'})
        MATCH (comp)-[:DEVELOPS]->(model:Model {kind: 'foundation'})
        MATCH (reg:Regulation {name: 'EU AI Act'})-[:APPLIES_IN]->(country)
        RETURN comp.name AS company, country.name AS country,
               collect(DISTINCT model.name) AS models,
               reg.name AS regulation
        ORDER BY company
    """).data()

    documents = session.run("""
        MATCH (comp:Company)-[:HEADQUARTERED_IN]->(country:Country {region: 'EU'})
        MATCH (comp)-[:DEVELOPS]->(:Model {kind: 'foundation'})
        MATCH (reg:Regulation {name: 'EU AI Act'})-[:APPLIES_IN]->(country)
        MATCH (d:Document)-[:MENTIONS]->(e)
        WHERE e = comp OR e = reg
        OPTIONAL MATCH (d)-[:HAS_KEYWORD]->(k:Keyword)
        RETURN DISTINCT d.title AS title, d.text AS text,
               collect(DISTINCT k.name) AS keywords
        ORDER BY title
    """).data()

    return knowledge, documents


def run_case(session, title, question, naive_terms, retrieve_fn):
    print("\n" + "=" * 72)
    print(title)
    print("Q:", question)

    naive_docs = naive_keyword_retrieve(session, naive_terms)
    print("\nnaive lexical hit (no relationships):")
    for row in naive_docs:
        print("-", row["title"], "| keywords:", row["keywords"])

    knowledge, documents = retrieve_fn(session)
    print("\ngraph retrieval — knowledge:")
    for row in knowledge:
        print("-", dict(row))
    print("graph retrieval — documents:")
    for row in documents:
        print("-", row["title"], "|", row.get("about_company", ""), "| keywords:", row["keywords"])

    context = format_context(knowledge, documents)
    answer = ask_llm(question, context)
    print("\nanswer:")
    print(answer)


def main():
    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(erase_graph)
        session.execute_write(build_knowledge_graph)
        session.execute_write(build_lexical_graph)
        print("knowledge + lexical graphs ready")

        # case 1: answer depends on COMPETES_WITH, which never appears in documents
        run_case(
            session,
            title="case 1 — multi-hop relation absent from text",
            question=(
                "What safety practices are described for US competitors of Mistral AI?"
            ),
            naive_terms=["Mistral", "competitor", "safety"],
            retrieve_fn=retrieve_us_competitors_of_mistral,
        )

        # case 2: structured filters (EU + foundation + regulation), then evidence docs
        run_case(
            session,
            title="case 2 — structured constraints + supporting documents",
            question=(
                "Which EU companies develop foundation models under the EU AI Act, "
                "and what compliance issues do the notes report?"
            ),
            naive_terms=["EU", "compliance", "foundation"],
            retrieve_fn=retrieve_eu_foundation_compliance,
        )


if __name__ == "__main__":
    main()
    driver.close()
