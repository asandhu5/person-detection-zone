import sys
import os
from unittest.mock import MagicMock


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mock_st = MagicMock()
mock_st.cache_resource = lambda func: func
sys.modules["streamlit"] = mock_st

sys.modules["ultralytics"] = MagicMock()
