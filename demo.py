
import pandas as pd
from common.config import InsightAgentConfig
from common.agent import run_insight_agent
from common.render import render_bullets

cur = pd.read_csv("sample_current.csv")
cfg = InsightAgentConfig(top_n=5, require_serviced=True)
payload = run_insight_agent(cur, previous_df=None, config=cfg)
print(render_bullets(payload, top_n=cfg.top_n))
