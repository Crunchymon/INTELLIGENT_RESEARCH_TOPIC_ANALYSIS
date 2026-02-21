import os

CORPUS = {
    "doc_1_inflation.txt": """The recent surge in inflation has forced central banks to tighten monetary policy. Interest rates have increased sharply in response to rising consumer prices and persistent supply chain disruptions. Policymakers are concerned that inflation expectations may become entrenched, leading to long-term economic instability. The central bank emphasized the need for price stability to protect purchasing power and maintain confidence in the financial system.""",

    "doc_2_inflation.txt": """High inflation continues to challenge the global economic system. Wage growth has not kept pace with rising food and energy prices. Monetary authorities are using contractionary policy tools, including higher benchmark interest rates and reduced liquidity, to stabilize the economy. Fiscal discipline is also being emphasized to reduce deficits and restore macroeconomic stability.""",

    "doc_3_inflation.txt": """The economic impact of sustained inflation affects households, businesses, and government institutions. Rising prices distort investment decisions and reduce real income. Policymakers must balance growth objectives with inflation control mechanisms. Central banks play a crucial role in regulating money supply and guiding expectations within the financial system.""",

    "doc_4_inflation.txt": """Persistent inflation poses structural risks to long-term economic growth. Interest rate hikes aim to slow demand and reduce overheating in the economy. However, aggressive monetary tightening may increase unemployment and reduce investment. A coordinated fiscal and monetary response is required to maintain systemic stability and ensure sustainable development.""",

    "doc_5_ai.txt": """Artificial intelligence systems are increasingly used in financial, healthcare, and defense sectors. Policymakers are developing regulatory frameworks to ensure transparency, accountability, and fairness. Ethical concerns include algorithmic bias, data privacy, and automated decision-making without human oversight. A balanced policy approach is required to promote innovation while protecting fundamental rights.""",

    "doc_6_ai.txt": """AI governance is becoming a global priority. Governments are proposing regulation to control high-risk AI applications. The economic impact of artificial intelligence includes productivity gains as well as labor market disruption. Regulatory systems must ensure safety standards, auditing mechanisms, and responsible deployment of advanced machine learning models.""",

    "doc_7_ai.txt": """The rapid expansion of artificial intelligence raises questions about accountability and systemic risk. Autonomous systems operating in critical infrastructure require strong oversight. Policymakers must coordinate internationally to prevent regulatory fragmentation. Ethical AI development requires transparency, fairness, and enforceable compliance standards.""",

    "doc_8_ai.txt": """Machine learning technologies are reshaping the economic landscape. However, algorithmic bias and opaque decision-making processes undermine trust in AI systems. Regulatory frameworks are being designed to reduce systemic harm while encouraging research and innovation. International cooperation is essential to manage cross-border AI governance challenges.""",

    "doc_9_climate.txt": """Climate change presents a systemic risk to the global economic system. Governments are implementing climate policy to reduce carbon emissions and accelerate the transition to renewable energy. Investment in sustainable infrastructure is essential to mitigate long-term environmental and economic damage.""",

    "doc_10_climate.txt": """The transition to clean energy requires coordinated regulatory frameworks and public investment. Carbon pricing mechanisms are being introduced to align market incentives with climate objectives. Policymakers emphasize sustainability and resilience in economic planning to address climate-related risks.""",

    "doc_11_climate.txt": """Global climate agreements aim to limit temperature increases and reduce greenhouse gas emissions. Energy systems must adapt through technological innovation and regulatory reform. The economic impact of climate change includes infrastructure damage, agricultural disruption, and migration pressures.""",

    "doc_12_climate.txt": """Sustainable development depends on balancing environmental protection with economic growth. Climate policy includes emissions regulation, renewable energy subsidies, and green financing mechanisms. International cooperation plays a critical role in ensuring effective implementation of climate strategies."""
}

def create_corpus(folder_name="sample_corpus"):
    os.makedirs(folder_name, exist_ok=True)

    for filename, content in CORPUS.items():
        path = os.path.join(folder_name, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Corpus created successfully in '{folder_name}' with {len(CORPUS)} documents.")

if __name__ == "__main__":
    create_corpus()