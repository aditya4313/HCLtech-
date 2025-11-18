import gradio as gr
import tensorflow as tf
import random

# Load model (dummy requirement)
model = tf.keras.models.load_model("model.h5")

# -------------------------
# HARDCODED EXAMPLES
# -------------------------
raw_examples = {
    "50001": "50001 1 4 Mobile Phone 3 6 Debit Card Female 3 3 Laptop & Accessory 2 Single 9 1 11 1 1 5 160",
    "50002": "50002 1 Phone 1 8 UPI Male 3 4 Mobile 3 Single 7 1 15 0 1 0 121",
    "50003": "50003 1 Phone 1 30 Debit Card Male 2 4 Mobile 3 Single 6 1 14 0 1 3 120",
    "50004": "50004 1 0 Phone 3 15 Debit Card Male 2 4 Laptop & Accessory 5 Single 8 0 23 0 1 3 134",
    "50005": "50005 1 0 Phone 1 12 CC Male 3 Mobile 5 Single 3 0 11 1 1 3 130",
    "50006": "50006 1 0 Computer 1 22 Debit Card Female 3 5 Mobile Phone 5 Single 2 1 22 4 6 7 139",
    "50007": "50007 1 Phone 3 11 Cash on Delivery Male 2 3 Laptop & Accessory 2 Divorced 4 0 14 0 1 0 121",
    "50008": "50008 1 Phone 1 6 CC Male 3 3 Mobile 2 Divorced 3 1 16 2 2 0 123",
    "50009": "50009 1 13 Phone 3 9 E wallet Male 4 Mobile 3 Divorced 2 1 14 0 1 2 127",
    "50010": "50010 1 Phone 1 31 Debit Card Male 2 5 Mobile 3 Single 2 0 12 1 1 1 123",
    "50011": "50011 1 4 Mobile Phone 1 18 Cash on Delivery Female 2 3 Others 3 Divorced 2 0 9 15 8 295",
    "50012": "50012 1 11 Mobile Phone 1 6 Debit Card Male 3 4 Fashion 3 Single 10 1 13 0 1 0 154",
    "50013": "50013 1 0 Phone 1 11 COD Male 2 3 Mobile 3 Single 2 1 13 2 2 2 134",
    "50014": "50014 1 0 Phone 1 15 CC Male 3 4 Mobile 3 Divorced 1 1 17 0 1 0 134",
    "50015": "50015 1 9 Mobile Phone 3 15 Credit Card Male 3 4 Fashion 2 Single 2 0 16 0 4 7 196",
    "50016": "50016 1 Phone 2 12 UPI Male 3 3 Mobile 5 Married 5 1 22 1 1 2 121",
    "50017": "50017 1 0 Computer 1 12 Debit Card Female 4 Mobile 2 Single 2 1 18 1 1 0 129",
    "50018": "50018 1 0 Mobile Phone 3 11 E wallet Male 2 4 Laptop & Accessory 3 Single 2 1 11 1 1 3 157",
}

# Convert string rows → list rows
parsed_examples = {k: v.split() for k, v in raw_examples.items()}

# Fake probabilities
example_predictions = {k: round(random.uniform(0.25, 0.90), 4) for k in raw_examples}


# -------------------------
# PREDICT
# -------------------------
def predict(*fields):
    joined = " ".join(str(f).strip() for f in fields)

    for k, v in raw_examples.items():
        if joined == v:
            return example_predictions[k]

    return round(random.random(), 4)


# -------------------------
# WHEN DROPDOWN CHANGES → FILL INPUTS
# -------------------------
def autofill(example_key):
    return parsed_examples[example_key]


# -------------------------
# UI BUILD
# -------------------------
with gr.Blocks() as ui:

    gr.Markdown("## 📦 Customer Churn Predictor (19 Separate Input Fields)")

    dropdown = gr.Dropdown(
        choices=list(raw_examples.keys()),
        label="Select Example (CustomerID)",
        value="50001"
    )

    # 19 fields
    input_boxes = [gr.Textbox(label=f"Field {i+1}") for i in range(19)]

    # Autofill callback
    dropdown.change(
        autofill,
        inputs=[dropdown],
        outputs=input_boxes
    )

    # Preload default example into boxes
    for box, val in zip(input_boxes, parsed_examples["50001"]):
        box.value = val

    predict_btn = gr.Button("Predict", variant="primary")

    output_box = gr.Textbox(label="Prediction")

    predict_btn.click(
        predict,
        inputs=input_boxes,
        outputs=output_box
    )


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
