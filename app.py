import base64
import subprocess
import logging
from pathlib import Path
import shutil
import pandas as pd
import oci
from dotenv import load_dotenv, find_dotenv
from pdf2image import convert_from_path
import gradio as gr
import os
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SlideSynthStudio")

# Read local .env file
_ = load_dotenv(find_dotenv())

# Configuration
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def convert_pptx_to_pdf(input_path: str, output_dir: str) -> str:
    """
    Convert PPTX to PDF using LibreOffice
    Returns the path of the generated PDF file
    """
    try:
        logger.info(f"Starting conversion to PDF: {Path(input_path).name}")
        logger.debug(f"Input path: {input_path}")
        logger.debug(f"Output directory: {output_dir}")

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # LibreOffice command to convert pptx to pdf
        pdf_output_path = Path(output_dir) / (Path(input_path).stem + ".pdf")

        cmd = [
            'soffice',
            '--headless',  # Run in headless mode (no GUI)
            '--convert-to', 'pdf',  # Convert to PDF
            '--outdir', output_dir,  # Output directory for the PDF
            input_path  # Input PPTX file
        ]

        # Execute the conversion command
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        if pdf_output_path.exists():
            logger.info(f"Successfully converted to PDF: {pdf_output_path}")
            return str(pdf_output_path)
        else:
            logger.error(f"PDF conversion failed for {input_path}")
            return ""

    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed for {input_path}")
        logger.error(f"Error details: {e.stderr}")
        raise RuntimeError(f"PPT to PDF conversion error: {e.stderr}")
    except Exception as e:
        logger.exception(f"Unexpected error during conversion: {str(e)}")
        raise


def convert_pdf_to_images(pdf_path: str, output_dir: str) -> bool:
    """
    Convert PDF pages to PNG images using pdf2image
    """
    try:
        logger.info(f"Starting PDF to images conversion: {pdf_path}")

        # Convert PDF to images (one per page)
        images = convert_from_path(pdf_path, dpi=300, thread_count=4)

        # Save each page as a PNG image
        for i, image in enumerate(images):
            image_filename = Path(output_dir) / f"page_{i + 1}.png"
            image.save(image_filename, 'PNG')
            logger.debug(f"Saved image: {image_filename}")

        return True

    except Exception as e:
        logger.error(f"PDF to image conversion failed: {str(e)}")
        return False


def process_ppt(file_tuple: tuple, progress: gr.Progress = gr.Progress()) -> bool:
    """Process a single PPT file"""
    file_path, filename = file_tuple
    output_subdir = Path(OUTPUT_DIR) / Path(filename).stem

    try:
        logger.info(f"Processing started: {filename}")
        progress(0, desc=f"Initializing {filename}")

        # Create output directory
        output_subdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory: {output_subdir}")

        # Convert PPTX to PDF
        pdf_path = convert_pptx_to_pdf(
            input_path=file_path,
            output_dir=str(output_subdir)
        )

        if not pdf_path:
            logger.error(f"PPT to PDF conversion failed for {filename}")
            return False

        # Convert PDF to images
        if not convert_pdf_to_images(pdf_path, str(output_subdir)):
            logger.error(f"PDF to images conversion failed for {filename}")
            return False

        progress(1.0, desc=f"Completed {filename}")
        logger.info(f"Successfully processed: {filename}")
        return True

    except Exception as e:
        logger.error(f"Processing failed: {filename} - {str(e)}")
        try:
            progress(1.0, desc=f"Failed: {filename}")
        except Exception as progress_error:
            logger.error(f"Progress update failed: {progress_error}")
        return False


def batch_process(files: list) -> dict:
    """Batch process multiple PPT files"""
    logger.info(f"Starting batch processing of {len(files)} files")
    file_tuples = [(file.name, os.path.basename(file.name)) for file in files]

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_ppt, file_tuples))

        success_count = sum(results)
        failed_count = len(results) - success_count
        logger.info(f"Batch completed - Success: {success_count}, Failed: {failed_count}")

        return {"visible": True, "value": f"Processed {success_count}/{len(files)} files successfully"}

    except Exception as e:
        logger.exception(f"Batch processing failed: {str(e)}")
        return {"visible": True, "value": f"Processing failed: {str(e)}"}


def preview_file(file):
    """
    Preview a PPT/PPTX or PDF file by converting it to images and returning the image paths.
    """
    try:
        logger.info(f"Starting preview for file: {file.name}")

        # Create a temporary output directory for preview
        preview_output_dir = Path(OUTPUT_DIR) / "preview"
        preview_output_dir.mkdir(parents=True, exist_ok=True)

        # Clear previous preview files
        for existing_file in preview_output_dir.glob("*"):
            existing_file.unlink()

        # Check file type
        file_extension = Path(file.name).suffix.lower()

        if file_extension == ".pdf":
            # If the file is a PDF, directly convert it to images
            pdf_path = file.name
        elif file_extension in [".ppt", ".pptx"]:
            # If the file is a PPT/PPTX, convert it to PDF first
            pdf_path = convert_pptx_to_pdf(
                input_path=file.name,
                output_dir=str(preview_output_dir)
            )
            if not pdf_path:
                logger.error(f"PPT to PDF conversion failed for preview: {file.name}")
                return []
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return []

        # Convert PDF to images
        if not convert_pdf_to_images(pdf_path, str(preview_output_dir)):
            logger.error(f"PDF to images conversion failed for preview: {file.name}")
            return []

        # Get the list of generated image files
        image_files = sorted(preview_output_dir.glob("*.png"))
        image_paths = [str(image) for image in image_files]

        logger.info(f"Preview images generated: {len(image_paths)}")
        return image_paths

    except Exception as e:
        logger.error(f"Preview failed: {str(e)}")
        return []


def process_image_with_ai(image_path: str) -> str:
    """
    Process an image using OCI Generative AI Service and return the response text.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: AI-generated response text.
    """
    # Fixed user prompt
    user_prompt = """
    You are an expert who extracts images, charts, and text and explains them while maintaining the original language.

    ## Instructions

    - Given the input, extract the charts, images, and tables and provide a detailed explanation of what the charts/images/tables are trying to convey.
    - Provide a description of each image/chart/table/text separately.
    - Ensure to include the coordinates of the image/charts/tables extracted from the page as output in each section.

    The output should be in the following format with each section header prefixed with ###:

    ### Charts:
        charts_explanation
        chart_coordinates

        note:
        - chart_explanation is a detailed explanation of the charts/graphs.
        - There can be more than one chart or graph. Explain each chart/graph separately.
        - Pay attention to the header above each chart to interpret what the chart is about.
        - Pay special attention to legends in the chart to interpret what each graph inside the chart indicates.
        - Interpret the x-axis and y-axis based on the label given for each axis.
        - Provide a detailed explanation of what the chart is trying to convey in English.
        - chart_coordinates is the precise coordinates of each chart/graph on the page along with the page number.
        - The output is only a single string "NOT FOUND" enclosed by ``` if there are no charts or graphs found.

    ### Tables:
        extracted_table
        table_explanation
        table_coordinates

        note:
        - extracted_table is the table extracted from the page as-is with original content and language.
        - Extract each table separately.
        - table_explanation is the detailed explanation of the table.
        - table_coordinates is the precise coordinates of each table on the page along with the page number.
        - Each extracted_table should be followed by its corresponding table_explanation.
        - The output is only a single string "NOT FOUND" enclosed by ``` if there are no tables found.

    ### Flowcharts:
        flowchart_explanation
        flowchart_coordinates

        note:
        - flowchart_explanation is the detailed explanation of the flowchart.
        - Provide a verbose and detailed explanation of what the chart is trying to convey in English, along with numbers and percentages if any.
        - flowchart_coordinates is the precise coordinates of each flowchart on the page along with the page number.
        - The output is only a single string "NOT FOUND" enclosed by ``` if there are no flowcharts found.

    ### Other Images:
        image_explanation

        note:
        - image_explanation is the detailed explanation of each image other than tables, charts, and flowcharts. It is 'NO' if no other image is found.
        - Pay attention to the header, footer, and notes of the image.
        - The output is only a single string "NOT FOUND" enclosed by ``` if there are no other images found.

    ### Extracted Text:
        extracted_text

        note:
        - extracted_text is the original text as-is extracted from the page.
        - Extract all the text present on the page and output the extracted text enclosed in ```.
        - Do not summarize the text.
    """

    # Read the image file and encode it as Base64
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to read or encode image: {str(e)}")
        return f"Error: Failed to process image - {str(e)}"

    # Setup basic variables
    compartment_id = os.environ["OCI_COMPARTMENT_ID"]
    CONFIG_PROFILE = "DEFAULT"
    config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

    # Service endpoint
    endpoint = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com"

    # Initialize Generative AI client
    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )

    # Prepare chat request
    chat_detail = oci.generative_ai_inference.models.ChatDetails()

    # Set the user prompt and image path as input
    text_content = oci.generative_ai_inference.models.TextContent()
    text_content.text = f"{user_prompt}\n\nImage Path: {image_path}"
    image_content = oci.generative_ai_inference.models.ImageContent(
        type="IMAGE",
        image_url={
            "url": f"data:image/png;base64,{base64_image}"  # Assuming PNG format
        }
    )

    message = oci.generative_ai_inference.models.Message()
    message.role = "USER"
    message.content = [text_content, image_content]

    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.messages = [message]
    chat_request.max_tokens = 3000
    chat_request.temperature = 0.5
    chat_request.frequency_penalty = 0
    chat_request.presence_penalty = 0
    chat_request.top_p = 0.75
    chat_request.top_k = -1

    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        model_id="meta.llama-3.2-90b-vision-instruct"
    )
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = compartment_id

    # Send request to Generative AI Service
    chat_response = generative_ai_inference_client.chat(chat_detail)
    logger.info(f"Chat response: {chat_response.data}")

    # Extract and return the response text
    response_text = chat_response.data.chat_response.choices[0].message.content[0].text
    return response_text


def process_gallery_images(gallery_images: list) -> pd.DataFrame:
    """
    Process all images in the gallery using process_image_with_ai and return a DataFrame with results.
    """
    results = []
    for image_tuple in gallery_images:
        # Extract image path from the tuple (image_path, label)
        image_path = image_tuple[0]  # image_path is the first element of the tuple
        image_name = Path(image_path).name
        try:
            response = process_image_with_ai(image_path)
            results.append({"Image Name": image_name, "AI Generated Image Text": response})
        except Exception as e:
            logger.error(f"Error processing image {image_name}: {str(e)}")
            results.append({"Image Name": image_name, "AI Generated Image Text": f"Error: {str(e)}"})

    return pd.DataFrame(results)


# Function to generate a voice script from the AI-extracted content
def generate_voice_script(input_text):
    # Setup basic variables
    compartment_id = os.environ["OCI_COMPARTMENT_ID"]
    CONFIG_PROFILE = "DEFAULT"
    config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

    # Service endpoint
    endpoint = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com"

    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )
    chat_detail = oci.generative_ai_inference.models.ChatDetails()

    # Define the prompt for the voice script processing
    prompt = (
        "Process the content field from the following input and convert it into a script suitable for voice recording. "
        "Return the processed text without any additional explanations. The content should meet the following requirements:\n"
        "1. For any English abbreviations or technical terms, expand them to their full forms.\n"
        "2. Remove all Markdown formatting such as asterisks, hash symbols, etc.\n"
        "3. Eliminate any line breaks or paragraph separators.\n"
        "4. Break up complex and long sentences into shorter phrases to make the script more conversational.\n"
        "5. The content should be prepared for use in a spoken script.\n\n"
        "Provide the processed output as plain text. Do not include any extra comments, explanations, or formatting.\n\n"
        f"{input_text}"
    )

    chat_request = oci.generative_ai_inference.models.CohereChatRequest()
    chat_request.preamble_override = "You are an expert in generating scripts for voice recordings."
    chat_request.message = prompt
    chat_request.max_tokens = 3000
    chat_request.temperature = 0.5
    chat_request.frequency_penalty = 0
    chat_request.top_p = 0.75
    chat_request.top_k = 0

    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        model_id="cohere.command-r-plus-08-2024"
    )
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = compartment_id

    chat_response = generative_ai_inference_client.chat(chat_detail)
    logger.info(f"Chat Response: {chat_response.data}")

    # Extracting the voice script result from the response
    voice_script = chat_response.data.chat_response.text if chat_response.data else "No result returned."
    logger.info(f"Voice script: {voice_script}")

    return voice_script


def generate_voice_scripts_for_responses(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process each 'AI Response' and generate a voice script, returning a new dataframe with the added 'Voice Script' column.
    """
    # Applying the generate_voice_script function to each row in the 'AI Response' column
    results_df['AI Generated Voice Script'] = results_df['AI Generated Image Text'].apply(generate_voice_script)
    scripts_df = results_df[['Image Name', 'AI Generated Voice Script']]
    return scripts_df


def translate_text(input_text, target_language="ja"):
    """
    Translate the input text to the target language using OCI Generative AI.

    Args:
        input_text (str): The text to be translated.
        target_language (str): The language to translate the input text into. Defaults to English ("en").

    Returns:
        str: The translated text.
    """
    # Setup basic variables
    compartment_id = os.environ["OCI_COMPARTMENT_ID"]
    CONFIG_PROFILE = "DEFAULT"
    config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)

    # Service endpoint (Assumed to be for translation service)
    endpoint = "https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com"

    # Initialize the OCI client for Generative AI
    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=endpoint,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )
    chat_detail = oci.generative_ai_inference.models.ChatDetails()

    # Define the prompt for translation
    prompt = (
        f"Translate the following text into {target_language}, "
        f"ensuring that any technical terms or professional jargon are kept in their original form and not translated.\n"
        f"Input text: {input_text}\n"
        f"Return only the translated text, without any additional comments, explanations, or formatting."
    )

    # Prepare the request to the Generative AI service
    chat_request = oci.generative_ai_inference.models.CohereChatRequest()
    chat_request.preamble_override = (
        f"You are an expert translator with the ability to translate text into {target_language}. \n"
        f"Your task is to accurately translate the input text while preserving any technical terms, brand names, "
        f"or professional jargon in their original form. \n"
        f"Ensure the translation is grammatically correct, contextually appropriate, and sounds natural in {target_language}. \n"
        f"Based on the context, adjust the tone of the translation to be either formal or casual as needed, "
        f"while maintaining the integrity of the message. \n"
        f"Do not include any explanations, comments, or additional formatting. Provide only the translated text."
    )
    chat_request.message = prompt
    chat_request.max_tokens = 3000
    chat_request.temperature = 0.5
    chat_request.frequency_penalty = 0
    chat_request.top_p = 0.75
    chat_request.top_k = 0

    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        model_id="cohere.command-r-plus-08-2024"
    )
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = compartment_id

    try:
        # Send request to OCI Generative AI service
        chat_response = generative_ai_inference_client.chat(chat_detail)
        logger.info(f"Chat Response: {chat_response.data}")

        # Extract and return the translated text from the response
        translated_text = chat_response.data.chat_response.text if chat_response.data else "No result returned."
        logger.info(f"Translated text: {translated_text}")
        return translated_text
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return f"Error: {str(e)}"


def generate_translated_scripts_for_responses(scripts_df: pd.DataFrame) -> pd.DataFrame:
    scripts_df['AI Generated Translated Script'] = scripts_df['AI Generated Voice Script'].apply(translate_text)
    translated_scripts_df = scripts_df[['Image Name', 'AI Generated Translated Script']]
    return translated_scripts_df


def generate_download_file(translated_scripts_df: pd.DataFrame, scripts_df: pd.DataFrame, results_df: pd.DataFrame):
    filepath = '/tmp/result.xlsx'
    with pd.ExcelWriter(filepath) as writer:
        translated_scripts_df.to_excel(writer, index=False, sheet_name='Sheet1')
        scripts_df.to_excel(writer, index=False, sheet_name='Sheet2')
        results_df.to_excel(writer, index=False, sheet_name='Sheet3')
    return gr.DownloadButton(value=filepath, visible=True)


theme = gr.themes.Default(
    primary_hue="red",
    secondary_hue="indigo",
    neutral_hue="gray",
    spacing_size="sm",
    font=[gr.themes.GoogleFont('Roboto'), gr.themes.GoogleFont('Noto Sans SC'), gr.themes.GoogleFont('Noto Sans JP'),
          gr.themes.GoogleFont('sans-serif')],
    font_mono=[gr.themes.GoogleFont('IBM Plex Mono'), gr.themes.GoogleFont('ui-monospace'),
               gr.themes.GoogleFont('Consolas'), gr.themes.GoogleFont('monospace')],
)

# Gradio UI Configuration
with gr.Blocks(title="Slide Synth Studio", theme=theme) as app:
    gr.Markdown("# 📊 Slide Synth Studio")

    with gr.Tab("Preview"):
        with gr.Row():
            file_input_preview = gr.File(
                file_types=[".pdf", ".ppt", ".pptx"],  # Allow PDF and PPT/PPTX files
                label="Select PDF or PPT for Preview"
            )
        with gr.Row():
            preview_btn = gr.Button("Generate File Preview", variant="primary")

        with gr.Row():
            gallery = gr.Gallery(
                label="Preview Thumbnails",
                columns=3,
                object_fit="contain",
                height="auto"
            )

        with gr.Row():
            process_images_btn = gr.Button("Generate Image Texts by AI", variant="primary")
        with gr.Row():
            results_df = gr.Dataframe(
                label="AI Processing Results",
                headers=["Image Name", "AI Generated Image Text"],
                datatype=["str", "str"],
                column_widths=[2, 8],
                interactive=False,
                wrap=True,
            )

        with gr.Row():
            generate_voice_scripts_btn = gr.Button("Generate Voice Scripts by AI", variant="primary")
        with gr.Row():
            scripts_df = gr.Dataframe(
                label="AI Processing Results",
                headers=["Image Name", "AI Generated Voice Script"],
                datatype=["str", "str"],
                column_widths=[2, 8],
                interactive=False,
                wrap=True,
            )

        with gr.Row():
            generate_translated_scripts_btn = gr.Button("Generate Translated Scripts by AI", variant="primary")
        with gr.Row():
            translated_scripts_df = gr.Dataframe(
                label="AI Processing Results",
                headers=["Image Name", "AI Generated Translated Script"],
                datatype=["str", "str"],
                column_widths=[2, 8],
                interactive=False,
                wrap=True,
            )

        with gr.Row():
            download_btn = gr.DownloadButton("Download Voice Scripts and Image Texts", variant="primary")

    with gr.Tabs():
        with gr.Tab("Processing", visible=False):
            with gr.Row():
                file_input = gr.File(
                    file_count="multiple",
                    file_types=[".pptx"],
                    label="Upload PPT Files"
                )
            with gr.Row():
                process_btn = gr.Button("Start Processing", variant="primary")

            with gr.Row():
                process_status = gr.Textbox(
                    label="Processing Status",
                    visible=False,
                    interactive=False
                )

    # Event handlers
    process_btn.click(
        fn=batch_process,
        inputs=file_input,
        outputs=process_status
    )

    # Preview tab event handlers
    preview_btn.click(
        lambda: gr.Gallery(value=None),
        inputs=[],
        outputs=[gallery]
    ).then(
        fn=preview_file,  # Use the updated preview_file function
        inputs=file_input_preview,
        outputs=[gallery]
    )

    # Process images with AI
    process_images_btn.click(
        lambda: gr.Dataframe(value=None),  # Clear the results dataframe
        inputs=[],
        outputs=[results_df]
    ).then(
        fn=process_gallery_images,
        inputs=[gallery],
        outputs=[results_df]
    )

    # When the button is clicked, generate the voice scripts
    generate_voice_scripts_btn.click(
        lambda: gr.Dataframe(value=None),  # Clear the results dataframe
        inputs=[],
        outputs=[scripts_df]
    ).then(
        fn=generate_voice_scripts_for_responses,  # Function to generate voice scripts for each AI Response
        inputs=[results_df],  # Input: the dataframe with AI Responses
        outputs=[scripts_df]  # Output: the same dataframe, but with a new 'Voice Script' column
    )

    generate_translated_scripts_btn.click(
        lambda: gr.Dataframe(value=None),  # Clear the results dataframe
        inputs=[],
        outputs=[translated_scripts_df]
    ).then(
        fn=generate_translated_scripts_for_responses,  # Function to generate voice scripts for each AI Response
        inputs=[scripts_df],  # Input: the dataframe with AI Responses
        outputs=[translated_scripts_df]  # Output: the same dataframe, but with a new 'Voice Script' column
    ).then(
        fn=generate_download_file,
        inputs=[translated_scripts_df, scripts_df, results_df],
        outputs=[download_btn]
    )

app.queue()

if __name__ == "__main__":
    app.launch(
        server_port=7860,
        max_threads=8,
        share=False,
        show_error=True,
    )
