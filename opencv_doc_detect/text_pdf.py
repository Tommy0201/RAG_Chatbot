from fpdf import FPDF

def text_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text, align='L')  # 0 means auto width, 10 is the height of each line
    return pdf
  
if __name__ == "__main__":
    pdf= text_to_pdf("ce of residence life 295, v a superlative award for the chef this award. is presented to. 25.")
    