import inspect
import json
import re
import subprocess
import sys

import pdfkit
import requests
from slugify import slugify

###################################################
# UTILITIES FUNCTIONS
###################################################


def func_with_logging(func):
    def func_wrapper(*args, **kwargs):
        # https://stackoverflow.com/questions/34832573/python-decorator-to-display-passed-and-default-kwargs
        func_args_specs = inspect.getfullargspec(func)
        positional_args_count = len(func_args_specs.args) - len(
            func_args_specs.defaults
        )
        log_message = kwargs.get(
            "log_message",
            func_args_specs.defaults[
                func_args_specs.args.index("log_message") - positional_args_count
            ],
        )
        print("-" * 80)
        print(log_message)
        results = func(*args, **kwargs)
        print("-" * 80)
        return results

    return func_wrapper


def make_markdown_table(rows):
    # https://gist.github.com/m0neysha/219bad4b02d2008e0154?permalink_comment_id=4201220#gistcomment-4201220
    """the same input as above"""
    nl = "\n"
    markdown = nl
    markdown += f"| {' | '.join(rows[0])} |"
    markdown += nl
    markdown += f"| {' | '.join(['---']*len(rows[0]))} |"
    markdown += nl
    for entry in rows[1:]:
        markdown += f"| {' | '.join(entry)} |{nl}"
    return markdown


###################################################
# DEPLOYMENT FUNCTIONS
###################################################


@func_with_logging
def execute_command(command, log_message="executing command"):
    status = subprocess.run(command, shell=True, capture_output=True)
    if status.returncode != 0:  # A non-zero (1-255) exit status indicates failure.
        print(
            f"error while running {command} command.",
            "\n",
            "Automation failed! Please try it from the command line to investigate.",
        )
        print("shell stdout:", status.stdout)
        print("shell stderr:", status.stderr)
        sys.exit()


def convert_notebook_to_format(format, notebook_title, file_extension=None):
    if not file_extension:
        file_extension = format
    try:
        command = rf'''jupyter nbconvert --to {format}\
                        "notebooks/{notebook_title}/notebook_no_metadata.ipynb"\
                        --output notebook.{file_extension}\
                        --NbConvertApp.output_files_dir "assets/{notebook_title}"\
                        --output-dir "notebooks/{notebook_title}"'''
        execute_command(
            command=command,
            log_message=f"Convert notebook to {format.upper()}",
        )
    except Exception as e:
        print(e)
        print(
            "Could not convert the file. Please make sure the file is publicly accessible through the link.",
            "To check that, see if the file can be opened in a private (incognito) browser?",
        )


@func_with_logging
def get_notebook_file(
    notebook_title,
    notebook_file_id,
    log_message="Getting the notebook file...",
):
    with open(f"notebooks/{notebook_title}/notebook.ipynb", "wb+") as notebook_file:
        # https://stackoverflow.com/a/66771478/4412324
        notebook_file.write(
            requests.get(
                f"https://docs.google.com/uc?export=download&id={notebook_file_id}"
            ).content
        )


def add_markdown_metadata(
    notebook_title,
    notebook_data,
    log_message="add markdown metadata to the markdown file",
    accepted_keys=(
        "title",
        "author",
        "categories",
        "tags",
        "math",
        "mermaid",
    ),
):
    with open(f"notebooks/{notebook_title}/notebook.md", "r") as notebook_file:
        file_content = notebook_file.read()
    metadata_str = "\n".join(
        f"{key}: {value}"
        for key, value in notebook_data.items()
        if key in accepted_keys
    )
    metadata_str = "---\n" + metadata_str + "\n---\n\n"
    with open(f"notebooks/{notebook_title}/notebook.md", "w") as file:
        file.write(metadata_str)
        file.write(file_content)


@func_with_logging
def convert_to_pdf(
    input_path,
    output_path,
    log_message="Generate the PDF file from HTML",
):
    return pdfkit.from_file(input_path, output_path)


@func_with_logging
def generate_notebooks_table_in_readme(
    notebooks_data,
    accepted_keys=(
        "title",
        "description",
    ),
    log_message="Generate Notebooks table in READM.md",
):
    readme_content = open("README.md").read().strip()

    readme_content = readme_content.split("# Notebooks")[0].strip()

    # generate rows
    rows = list()
    for notebook_data in notebooks_data:
        row = [
            notebook_data[key] for key in notebook_data.keys() if key in accepted_keys
        ]
        rows.append(row)

    # generate headers
    rows.insert(0, accepted_keys)
    with open("README.md", "w") as file:
        file.write(readme_content)
        file.write("\n\n# Notebooks\n")
        rows_str = make_markdown_table(rows=rows)
        file.write(rows_str)


@func_with_logging
def replace_images_urls_in_markdown(
    notebook_title, log_message="Fix asset issue in markdown file"
):
    with open(f"notebooks/{notebook_title}/notebook.md", "r") as file:
        file_content = file.read()
    file_content = file_content.replace("](assets/", "](/assets/")
    with open(f"notebooks/{notebook_title}/notebook.md", "w") as file:
        file.write(file_content)


@func_with_logging
def replace_http_with_https(
    notebook_title,
    log_message="replace http with https to ensure https connection",
):
    with open(f"notebooks/{notebook_title}/notebook.html", "r") as file:
        file_content = file.read()
    file_content = file_content.replace("http://", "https://")
    with open(f"notebooks/{notebook_title}/notebook.html", "w") as file:
        file.write(file_content)


###################################################
# DEPLOYMENT COMMANDS
###################################################

notebooks = json.load(open("notebooks.json"))

for notebook in notebooks:

    notebook_link = notebook["link"]
    notebook_file_id = (
        notebook_link[notebook_link.index("drive/") :].split("/")[1].split("?")[0]
    )
    notebook_title = notebook["title"]
    notebook_title_slugified = slugify(notebook_title)  # to escape special characters

    execute_command(
        command=rf'''mkdir -p "notebooks/{notebook_title_slugified}"''',
        log_message="Creating notebook directory",
    )

    get_notebook_file(
        notebook_title=notebook_title_slugified,
        notebook_file_id=notebook_file_id,
    )

    execute_command(
        command=f'''jq -M 'del(.metadata.widgets)' "notebooks/{notebook_title_slugified}/notebook.ipynb" > "notebooks/{notebook_title_slugified}/notebook_no_metadata.ipynb"''',
        log_message="remove colab metadata for save html conversion. make sure jq tool is installed",
    )

    convert_notebook_to_format(
        notebook_title=notebook_title_slugified,
        format="html",
    )

    replace_http_with_https(notebook_title=notebook_title_slugified)

    convert_to_pdf(
        input_path=f"notebooks/{notebook_title_slugified}/notebook.html",
        output_path=f"notebooks/{notebook_title_slugified}/notebook.pdf",
    )

    convert_notebook_to_format(
        notebook_title=notebook_title_slugified,
        format="markdown",
        file_extension="md",
    )

    add_markdown_metadata(
        notebook_title=notebook_title_slugified,
        notebook_data=notebook,
    )

    replace_images_urls_in_markdown(notebook_title=notebook_title_slugified)

    # https://stackoverflow.com/a/46346719/4412324
    execute_command(
        command=rf"""cp -RL "notebooks/{notebook_title_slugified}/assets/{notebook_title_slugified}" "assets" """,
        log_message="""copy notebook assets to assets directory""",
    )

    execute_command(
        command=rf"""mv "notebooks/{notebook_title_slugified}/notebook.md" "_posts/{notebook['publication_date']}-{notebook_title_slugified.replace(' ','-')}.md" """,
        log_message="""moving the markdown file to _posts directory""",
    )

generate_notebooks_table_in_readme(notebooks_data=notebooks)


# TODO update the README.md table
