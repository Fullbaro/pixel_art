import jinja2
import pdfkit

class PDF:

    def __init__(self, data):
        self.context = data

    def start(self):
        template_loader = jinja2.FileSystemLoader("./")
        template_env = jinja2.Environment(loader=template_loader)

        template = template_env.get_template('./assets/template.html')
        output_text = template.render(self.context)

        config = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')
        pdfkit.from_string(output_text, './user_guide.pdf', configuration=config, css='./assets/style.css', options={'quiet': '', 'enable-local-file-access': ''})

        # Optionally save as html
        with open('./assets/document.html', 'w') as f:
            f.write(output_text)

        print("Document created!")