import json
from pathlib import Path
from agent.schemas import ResearchInput
from agent.engine import evaluate
if __name__=='__main__':
    d=Path(__file__).resolve().parent
    result=evaluate(ResearchInput.model_validate_json((d/'sample_input.json').read_text()))
    (d/'sample_output.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
