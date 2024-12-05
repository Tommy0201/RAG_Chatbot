import easyocr
import cv2
import re 
from spellchecker import SpellChecker
from pytesseract import pytesseract, Output
from PIL import Image

def detect_pytesseract(input_path):
    text = pytesseract.image_to_string(Image.open(input_path))
    return text
    
    
def detect_easyocr(input_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(input_path,paragraph=False)
    confidence = 0
    paragraph = ""
    for res in result:
        paragraph = paragraph + (res[1]) + " "
        confidence += (res[2]/len(result))
    print("Confidence: ",confidence)
    # print("Paragraph: ",paragraph)
    # with open(output_path, 'w') as f:
    new_paragraph = ""
    words = paragraph.split()
    for i, word in enumerate(words):
        # if i>0 and i % 12 == 0:
        #     new_paragraph += "\n"
        new_paragraph = new_paragraph + " " + word
    return paragraph

def preprocess_text(text1, text2):
    # Step 1: Convert to lowercase
    text1, text2 = text1.lower(), text2.lower()
    
    text1, text2 = re.sub(r"[^a-zA-Z0-9\s.,']", " ", text1), re.sub(r"[^a-zA-Z0-9\s.,']", " ", text2)    
    
    #Correct OCR spacing issues
    text1, text2 = re.sub(r"\s+", " ", text1), re.sub(r"\s+", " ", text2)  
    
    sentences1, sentences2 = text1.split("."), text2.split(".")
    sentences1 = [sentence.strip() for sentence in sentences1 if sentence.strip()]  
    sentences2 = [sentence.strip() for sentence in sentences2 if sentence.strip()] 
    
    if len(sentences1) >= len(sentences2):
        out = sentences1
    else:
        out = sentences2
    # spell = SpellChecker()
    # corrected_sentences = []
    # for sentence in sentences:
    #     words = sentence.split()
    #     corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    #     corrected_sentences.append(" ".join(corrected_words))
        
    clean_text = ". ".join(out) + "."
    
    return clean_text

def text_detect(input_path):
    easyocr_text = detect_easyocr(input_path)
    pytesseract_text = detect_pytesseract(input_path)
    return preprocess_text(easyocr_text, pytesseract_text)

if __name__ == "__main__":
    out1 = (detect_easyocr("opencv_doc_detect/IMG_0254.jpg"))
    print(out1)
    out2 = preprocess_text(out1)
    print(out2)