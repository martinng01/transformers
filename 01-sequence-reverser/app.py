from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import TransformerReverser
from typing import Annotated
import torch
import yaml

app = FastAPI()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class ModelInput(BaseModel):
    sequence: Annotated[
        list[Annotated[int, Field(ge=1, le=99)]],
        Field(min_length=10, max_length=10)
    ]


model = TransformerReverser(
    vocab_size=config['training']['vocab_size'],
    d_model=config['model']['d_model'],
    seq_len=config['training']['seq_len'],
    num_heads=config['model']['num_heads'],
    num_layers=config['model']['num_layers']
)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()


@app.get("/")
async def root():
    return {'status': 'online'}


@app.post('/predict')
async def predict(data: ModelInput):
    try:
        input_tensor = torch.tensor([data.sequence]).long()

        with torch.no_grad():
            logits = model(input_tensor)
            prediction = torch.argmax(logits, dim=-1).squeeze(0).tolist()

        return prediction
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")
