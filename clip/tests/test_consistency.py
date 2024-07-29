import numpy as np
import pytest
import torch
from PIL import Image

import clip


@pytest.mark.parametrize('model_name', clip.available_models())
def test_consistency(model_name):
    device = "cpu"
    jit_model, transform = clip.load(model_name, device=device)
    py_model, _ = clip.load(model_name, device=device, jit=False)
    img = Image.open("CLIP.png")
    
    image = transform(img).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    #print(image.shape)
    with torch.no_grad():
        logits_per_image, _ = jit_model(image, text)
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        logits_per_image, _ = py_model(image, text)
        py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print(py_probs)

    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)

test_consistency('RN50')
