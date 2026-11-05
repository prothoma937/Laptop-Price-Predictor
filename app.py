import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open('laptop_pipeline.pkl','rb') as file:
    model=pickle.load(file)

def predict_price(
        brand, name, spec_rating, processor, CPU, Ram, Ram_type,
        ROM, ROM_type, GPU, display_size, resolution_width, 
        resolution_height, OS, warranty):
    input_df=pd.DataFrame([[
        brand, name, spec_rating, processor, CPU, Ram, Ram_type, ROM, ROM_type, 
        GPU, display_size, resolution_width, resolution_height, OS, warranty
    ]],
    columns=['brand', 'name', 'spec_rating', 'processor', 'CPU', 'Ram', 'Ram_type',
       'ROM', 'ROM_type', 'GPU', 'display_size', 'resolution_width',
       'resolution_height', 'OS', 'warranty'])

    prediction=model.predict(input_df)[0]
    return f"Estimated Laptop Price: {round(prediction, 2)}"
    #app interface
inputs = [
    gr.Textbox(label="Brand", placeholder="Enter Your Brand"),
    gr.Textbox(label="Laptop Name", placeholder="Enter Laptop Name"),
    gr.Number(label="Spec Rating", placeholder="Enter Spec Rating"),
    gr.Textbox(label="Processor", placeholder="Enter Processor"),
    gr.Textbox(label="CPU", placeholder="Enter CPU"),
    gr.Number(label="RAM (in GB)", placeholder="Enter RAM in GB"),
    gr.Textbox(label="RAM Type", placeholder="Enter RAM Type"),
    gr.Number(label="ROM / Storage (in GB)", placeholder="Enter Storage in GB"),
    gr.Textbox(label="ROM Type", placeholder="Enter ROM Type"),
    gr.Textbox(label="GPU", placeholder="Enter GPU"),
    gr.Number(label="Display Size (inches)", placeholder="Enter Display Size"),
    gr.Number(label="Resolution Width", placeholder="Enter Resolution Width"),
    gr.Number(label="Resolution Height", placeholder="Enter Resolution Height"),
    gr.Textbox(label="Operating System (OS)", placeholder="Enter OS"),
    gr.Number(label="Warranty (Years)", placeholder="Enter Warranty")
]

app = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs="text", 
    title="Laptop Price Predictor")

app.launch(share=True)

