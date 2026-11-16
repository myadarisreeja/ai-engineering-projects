import os


class Settings:
    """Runtime settings, all overridable through environment variables."""

    def __init__(self):
        self.db_path = os.getenv("AGENTNET_DB", "agentnet.db")
        self.top_k = int(os.getenv("AGENTNET_TOP_K", "50"))          # Stage 2 shortlist size
        self.dense_floor = float(os.getenv("AGENTNET_DENSE_FLOOR", "0.20"))  # cheap cut before rerank
        self.threshold = float(os.getenv("AGENTNET_THRESHOLD", "0.45"))      # Stage 3 final cut
        self.max_notify = int(os.getenv("AGENTNET_MAX_NOTIFY", "5"))
        self.admin_key = os.getenv("AGENTNET_ADMIN_KEY", "dev-admin-key")
        self.posts_per_minute = int(os.getenv("AGENTNET_POSTS_PER_MIN", "10"))
        self.comments_per_minute = int(os.getenv("AGENTNET_COMMENTS_PER_MIN", "30"))
        self.rerank_mode = os.getenv("AGENTNET_RERANK", "heuristic")  # heuristic | llm
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.llm_model = os.getenv("AGENTNET_LLM_MODEL", "claude-haiku-5-5")
        self.unverified_trust = float(os.getenv("AGENTNET_UNVERIFIED_TRUST", "0.7"))
        self.ignore_after_hours = float(os.getenv("AGENTNET_IGNORE_AFTER_HOURS", "24"))


settings = Settings()
