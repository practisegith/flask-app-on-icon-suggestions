import torch
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify, send_file, abort
import os

app = Flask(__name__)

model_name = "sentence-transformers/all-mpnet-base-v2"
sentence_tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

icons = {
    "home": r"C:\icon 2\icon images\homeicon.png",
    "search": r"C:\icon 2\icon images\search icon.png",
    "chat": r"C:\icon 2\icon images\icons8-chat-40.png",
    "upload": r"C:\icon 2\icon images\icons8-upload-50.png",
    "download": r"C:\icon 2\icon images\icons8-download-50.png",
    "pdf": r"C:\icon 2\icon images\icons8-pdf-50.png",
    "mail": r"C:\icon 2\icon images\icons8-mail-50.png",
    "database": r"C:\icon 2\icon images\icons8-database-50.png",
    "notification": r"C:\icon 2\icon images\icons8-notification-50.png",
    "bug": r"C:\icon 2\icon images\icons8-bug-50.png"
}

def get_embedding(text, tokenizer):
    encoded_input = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    return output.pooler_output.squeeze(0)

@app.route('/get_closest_icon', methods=['POST'])
def get_closest_icon():
    data = request.json
    user_input = data.get('icon_description', '').strip().lower()

    if not user_input:
        return jsonify({"error": "No icon description provided"}), 400

    user_embedding = get_embedding(user_input, sentence_tokenizer)

    icon_embeddings = {}
    for description in icons.keys():
        if description not in icon_embeddings:
            icon_embeddings[description] = get_embedding(description, sentence_tokenizer)

    max_similarity = -float('inf')
    closest_icon = None
    for description, icon_embedding in icon_embeddings.items():
        similarity = torch.nn.functional.cosine_similarity(user_embedding.unsqueeze(0), icon_embedding.unsqueeze(0))
        if similarity > max_similarity:
            max_similarity = similarity
            closest_icon = description

    if closest_icon:
        icon_path = icons[closest_icon]
        icon_path = os.path.normpath(icon_path)
        if os.path.exists(icon_path):
            return send_file(icon_path, mimetype='image/png')
        else:
            return jsonify({"error": f"Icon image not found: {icon_path}"}), 404
    else:
        return jsonify({"message": f"No similar icon found for '{user_input}'."})

if __name__ == "__main__":
    app.run(debug=True)


