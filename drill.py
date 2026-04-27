"""
Module 6 Week A — Core Skills Drill: Text Processing & NLP Basics

Complete the three functions below. Each function has a docstring
describing its inputs, outputs, and purpose.

Run your work: python drill.py
Test your work: the autograder runs automatically when you open a PR.
"""

import spacy

# تحميل موديل اللغة الإنجليزية
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text, stop_words):
    """
    المهمة 1: معالجة مقطع نصي مسبقاً
    تنظيف النص من علامات الترقيم، المساحات، وتحويله لأحرف صغيرة مع حذف كلمات التوقف.
    """
    # معالجة النص عبر خط أنابيب spaCy
    doc = nlp(text)
    
    # فلترة الرموز: استبعاد الترقيم والمساحات وكلمات التوقف
    cleaned_tokens = [
        token.text.lower() 
        for token in doc 
        if not token.is_punct and not token.is_space and token.text.lower() not in stop_words
    ]
    
    return cleaned_tokens


def extract_linguistic_annotations(text):
    """
    المهمة 2: استخراج التعليقات اللغوية
    استخراج نص الرمز، وسم أجزاء الكلام (POS)، وتسمية التبعية (Dependency).
    """
    doc = nlp(text)
    
    # بناء قائمة من الأرباع (Tuples) لكل رمز
    annotations = [
        (token.text, token.pos_, token.dep_) 
        for token in doc
    ]
    
    return annotations


def extract_entities(text):
    """
    المهمة 3: تشغيل NER وإخراج الصيغة
    استخلاص الكيانات المسماة وتصنيفاتها من النص.
    """
    doc = nlp(text)
    
    # استخراج النص والوسم الخاص بكل كيان (مثل GPE, ORG, DATE)
    entities = [
        (ent.text, ent.label_) 
        for ent in doc.ents
    ]
    
    return entities


if __name__ == "__main__":
    # نص العينة للاختبار
    sample = (
        "The IPCC released its latest report in Geneva on March 20, 2024. "
        "Dr. Ahmad presented findings on Jordan's climate adaptation strategy "
        "at the COP28 conference in Dubai."
    )
    
    # قائمة كلمات التوقف
    stop_words = {"the", "a", "an", "in", "on", "at", "its", "of", "and", "is"}

    # تشغيل المهمة 1
    tokens = preprocess_text(sample, stop_words)
    if tokens is not None:
        print(f"Cleaned tokens ({len(tokens)}): {tokens[:10]}...")

    # تشغيل المهمة 2
    annotations = extract_linguistic_annotations(sample)
    if annotations is not None:
        print(f"\nAnnotations ({len(annotations)} tokens):")
        for tok, pos, dep in annotations[:5]:
            print(f"  {tok:15s} {pos:8s} {dep}")

    # تشغيل المهمة 3
    entities = extract_entities(sample)
    if entities is not None:
        print(f"\nEntities ({len(entities)}):")
        for text, label in entities:
            print(f"  {text:25s} {label}")