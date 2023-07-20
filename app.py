# Streamlit App to use Amazon Rekognition API

import streamlit as st
import boto3
import os
from PIL import Image
import tempfile
from botocore.exceptions import ClientError
import logging
import zipfile
import io
from io import BytesIO
import base64
import shutil
import datetime
import re
from zipfile import ZipFile, ZIP_STORED
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from datetime import datetime
import threading
from multiprocessing import Pool
from googleapiclient import errors
import time
from google_auth_oauthlib.flow import InstalledAppFlow
import requests
import json
from google.oauth2.credentials import Credentials
import webbrowser
import random
from PIL import ImageOps, ExifTags
import uuid
import glob
from streamlit_javascript import st_javascript
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from googleapiclient.http import MediaFileUpload
import traceback
import pyheif

logging.basicConfig(level=logging.INFO)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
    page_title='SmartLabel',
    page_icon='camera'
)   
hide_streamlit_style = """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """ 
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set AWS details (replace with your own details)
AWS_REGION_NAME = 'us-east-2'
AWS_ACCESS_KEY = 'AKIARK3QQWNWXGIGOFOH'
AWS_SECRET_KEY = 'ClAUaloRIp3ebj9atw07u/o3joULLY41ghDiDc2a'

# Initialize the S3 client
s3 = boto3.client('s3',
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Initialize boto3 client for AWS Rekognition
client = boto3.client('rekognition',
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def save_file_locally(file, person_name):
    """
    Save the uploaded file to a local directory named after the person.

    :param file: File to save
    :param person_name: Name of the person. Used to create a directory for the person.
    :return: The path to the saved file
    """
    directory = os.path.join(os.getcwd(), collection_id, 'training_imgs', person_name)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path


def resize_image(image, basewidth):
    img = Image.open(image)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img

def delete_collection(collection_id):
    try:
        response = client.delete_collection(CollectionId=collection_id)
    except ClientError as e:
        # logging.error(e)
        st.error("Program ID not found")
        raise e
    
def create_collection(collection_id):
    try:
        response = client.create_collection(CollectionId=collection_id)
        pass
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
            pass
        else:
            logging.error(e)
            raise e
        
def upload_file_to_s3(file, bucket, object_name):
    """
    Upload a file to an S3 bucket

    :param file: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    try:
        s3.upload_fileobj(file, bucket, object_name)
    except ClientError as e:
        logging.error('Error uploading file to S3: %s', e)
        return False
    logging.info('File uploaded successfully to bucket %s with key %s', bucket, object_name)
    return True


# Define function to add faces to the collection
def add_faces_to_collection(bucket, photo, collection_id, external_image_id):
    response = client.index_faces(
        CollectionId=collection_id,
        Image={'S3Object':{'Bucket':bucket,'Name':photo}},
        ExternalImageId=external_image_id,
        MaxFaces=5,
        QualityFilter="AUTO",
        DetectionAttributes=['ALL']
    )

    return response['FaceRecords']

def find_matching_faces(photo, collection_id):
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(photo))

    # Attempt to detect faces in the photo
    faces = client.detect_faces(Image={'Bytes': photo})
    # print(f"Number of faces detected by detect_faces: {len(faces['FaceDetails'])}")
    
    # For each face, search for matches in the collection
    face_matches = []
    for i, face in enumerate(faces['FaceDetails']):
        bbox = face['BoundingBox']
        # Convert bbox to the format required by search_faces_by_image
        width, height = image.width, image.height
        left = int(bbox['Left']*width)
        top = int(bbox['Top']*height)
        face_width = int(bbox['Width']*width)
        face_height = int(bbox['Height']*height)

        # Add a buffer to the bounding box coordinates to make sure the entire face is included
        buffer = 20  # adjust this value as needed
        left = max(0, left - buffer)
        top = max(0, top - buffer)
        right = min(width, left + face_width + buffer)
        bottom = min(height, top + face_height + buffer)

        face_crop = image.crop((left, top, right, bottom))
        bytes_io = io.BytesIO()
        face_crop.save(bytes_io, format='JPEG')
        face_bytes = bytes_io.getvalue()

        try:
            response = client.search_faces_by_image(
                CollectionId=collection_id,
                Image={'Bytes': face_bytes},
                FaceMatchThreshold=75,
                MaxFaces=5
            )
            face_matches.extend([match['Face']['ExternalImageId'] for match in response['FaceMatches']])
            # print(f"Face {i+1}: matches found")
        except Exception as e:
            pass
            # print(f"Face {i+1}: error occurred - {str(e)}")

    # print(f"Number of face matches found: {len(face_matches)}")
    return face_matches

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    return href

# Define function to list faces in the collection
def list_faces_in_collection(collection_id):
    response = client.list_faces(CollectionId=collection_id)
    return [face['ExternalImageId'] for face in response['Faces']]

# Define function to delete a face from the collection
def delete_face_from_collection(collection_id, face_id):
    response = client.delete_faces(CollectionId=collection_id, FaceIds=[face_id])
    return response['DeletedFaces']

# Define function to add a training image to an existing person
def add_training_image_to_person(collection_id, person_name, image):
    # Upload the image to S3 bucket
    object_name = f'{person_name}_{len(list_faces_in_collection(collection_id)) + 1}.jpg'
    upload_success = upload_file_to_s3(image, 'giacomo-aws-bucket', object_name)
    if upload_success:
        add_faces_to_collection('giacomo-aws-bucket', object_name, collection_id, object_name)

def make_request_with_exponential_backoff(request):
    for n in range(0, 5):
        try:
            return request.execute()
        except Exception as e:
            if isinstance(e, errors.HttpError) and e.resp.status == 403:
                time.sleep((2 ** n) + random.random())
            else:
                raise e
    print("Request failed after 5 retries")
    try:
        return request.execute()
    except:
        return None

def sanitize_name(name):
    """Sanitize names to match AWS requirements for ExternalImageId"""
    # Remove everything after a hyphen, an underscore or ".jpg"
    name = re.sub(r' -.*|_.*|\.jpg|\.JPG|\.jpeg|\.JPEG|\.png|\.PNG|\.heic|\.HEIC', '', name)
    # Keep only alphabets, spaces, and hyphens
    name = re.sub(r'[^a-zA-Z \-]', '', name)
    # Replace hyphens and spaces with underscores
    name = re.sub(r'[- ]', '_', name)
    # Replace multiple underscores with a single underscore
    name = re.sub(r'[_]+', '_', name)
    # Remove leading underscore if it exists
    if name.startswith('_'):
        name = name[1:]
    return name

def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        print("not corrected")
        # In case of exceptions, the image is left as is
        pass
    return image

def consolidate_labels(collection_id):
    # Dictionary to store labels
    labels_dict = {}

    # Iterate over all text files in the labels directory
    for filename in glob.glob(f'{collection_id}/labels/*.txt'):
        with open(filename, 'r') as f:
            # Read the labels from the file
            line = f.readline().strip()
            if ': ' in line:
                image_name, persons = line.split(': ')
                persons_list = persons.split(', ') if persons else []
            else:
                image_name = line[:-1]  # remove the trailing colon
                persons_list = []

            # Append the labels to the dictionary
            for person in persons_list:
                if person not in labels_dict:
                    labels_dict[person] = []
                labels_dict[person].append(image_name)

        # Delete the file after processing
        os.remove(filename)

    # Identify the 'group' key (assuming it's there)
    group_key = 'Group Photos'  # Change as needed
    group_images = labels_dict.pop(group_key, None)

    # Now, create a new consolidated file and write the 'group: images' at the top
    with open(f'{collection_id}/labels.txt', 'w') as f:
        if group_images is not None:
            f.write(f'{group_key}: {", ".join(group_images)}\n\n')

        # Write the rest of the labels
        for person, images in labels_dict.items():
            f.write(f'{person}: {", ".join(images)}\n\n')

    st.session_state['download_zip_created'] = True

def list_collections(max_results=20):
    """
    Get a list of all collection IDs

    :param max_results: Maximum number of collection IDs to return
    :return: List of collection IDs
    """
    response = client.list_collections(MaxResults=max_results)
    collection_ids = response['CollectionIds']
    default_id = 'your-default-collection-id'
    if default_id in collection_ids:
        collection_ids.remove(default_id)
    return collection_ids

def process_file(file, service, folder_id, person_images_dict, group_photo_threshold, collection_id, person_folder_dict):
    st.write(f"{file['name']} started")
    try:
        # Initialize persons
        persons = []
        request = service.files().get_media(fileId=file['id'])
        # Download and process the image
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            _, done = downloader.next_chunk()

        if file['name'].endswith('.heic') or file['name'].endswith('.HEIC'):
            heif_file = pyheif.read(fh.getvalue())
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode)
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='JPEG')
            img.save('converted3.jpg')
            byte_img = byte_arr.getvalue()
            return byte_img
        else:  # This will cover both .jpg and .png files
            img_io = io.BytesIO(fh.getvalue())
            img = resize_image(img_io, 1000)
            if img.mode != 'RGB':  # Convert to RGB if not already
                img = img.convert('RGB')
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='JPEG')
            byte_img = byte_arr.getvalue()

        detected_persons = find_matching_faces(byte_img, collection_id)

        if len(set(detected_persons)) >= group_photo_threshold:
            person_images_dict['Group Photos'].append(file['name'])
            persons = ['Group Photos']
        else:
            persons = detected_persons

        for person in set(persons):
            if person not in person_images_dict:
                person_images_dict[person] = []
            person_images_dict[person].append(file['name'])

            # Get the person's folder from the dictionary
            folder = person_folder_dict[person]

            current_year = datetime.now().year
            new_file_name = f"{person}_{current_year}_{file['name']}"
            copied_file = make_request_with_exponential_backoff(service.files().copy(fileId=file['id'], body={"name": new_file_name, "parents": [folder['id']]}))


    except Exception as e:
        print(f"{file['name']} threw an error: {e}")

    # Generate a unique filename using uuid library
    unique_filename = str(uuid.uuid4()) + '.txt'
    with open(f'{collection_id}/labels/{unique_filename}', 'w') as f:
        # Write the image name and persons detected to the file
        f.write(f"{file['name']}: {', '.join(set(persons))}")

    print(f"{file['name']}: {', '.join(set(persons))}")

def convert_heic_to_jpeg(file_name):
    # Full path to the original HEIC file
    original_file_path = os.path.abspath(file_name)

    # Read HEIC file
    heif_file = pyheif.read(original_file_path)
    
    # Convert to PIL Image
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    # Save as JPEG
    jpeg_filename = f'{uuid.uuid4()}.jpg' 
    image.save(jpeg_filename, 'JPEG')

    # Read the saved jpeg file and convert it to bytes
    with open(jpeg_filename, 'rb') as img_file:
        byte_img = img_file.read()

    # Delete both the original .heic file and converted .jpg file
    os.remove(original_file_path)
    os.remove(jpeg_filename)

    return byte_img



def process_folder(folder, service, interns_without_training_data, collection_id, parent_folder):
    has_training_image = False
    temp_file_path = None
    unique_filename = None

    query = f"'{folder['id']}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    intern_images = service.files().list(q=query).execute().get('files', [])

    query = f"'{parent_folder}' in parents and name='Training Images' and trashed = false"
    results = service.files().list(q=query).execute().get('files', [])

    if results:
        training_images_folder_id = results[0]['id']
    else:
        file_metadata = {
            'name': 'Training Images',
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder]
        }
        training_images_folder = service.files().create(body=file_metadata, fields='id').execute()
        training_images_folder_id = training_images_folder['id']

    for img in intern_images:
        try:
            if 'bio' in img['name'].lower():
                image_id = img['id']
                image_name = img['name']
                request = service.files().get_media(fileId=image_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    _, done = downloader.next_chunk()
 
                    
                if image_name.endswith('.heic') or image_name.endswith('.HEIC'):
                    # Save the HEIC file to the local directory first
                    unique_heic_filename = f'{uuid.uuid4()}.heic'
                    fh = io.BytesIO()
                    request = service.files().get_media(fileId=image_id)
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        _, done = downloader.next_chunk()

                    # Now fh holds the file content
                    with open(unique_heic_filename, 'wb') as f:  # Open a file in binary mode for writing
                        f.write(fh.getvalue())  # Write the content

                    # Then convert the saved HEIC file to JPEG
                    byte_img = convert_heic_to_jpeg(unique_heic_filename)

                else:
                    img_io = io.BytesIO(fh.getvalue())
                    img = resize_image(img_io, 1000)
                    if img.mode != 'RGB':  # Convert to RGB if not already
                       img = img.convert('RGB')
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format='JPEG')
                    byte_img = byte_arr.getvalue()

                intern_name = folder['name']
                print(intern_name)
                sanitized_intern_name = sanitize_name(intern_name)
                print(sanitized_intern_name)
                if sanitized_intern_name not in list_faces_in_collection(collection_id):
                    upload_success = upload_file_to_s3(io.BytesIO(byte_img), 'giacomo-aws-bucket', sanitized_intern_name)
                    if upload_success:
                        print(add_faces_to_collection('giacomo-aws-bucket', sanitized_intern_name, collection_id, sanitized_intern_name))
                        print(f'Person {sanitized_intern_name} added successfully')
                    else:
                        print('Failed to upload image')
                else:
                    print(f'{sanitized_intern_name} already in system')

                # After processing the image and saving to byte_img:

                # Copy the original image to 'Training Images' folder in Google Drive
                file_extension = os.path.splitext(image_name)[1]  # Extracting the file extension from the original name
                file_metadata = {
                    'name': f'{sanitized_intern_name}{file_extension}',  # Using sanitized name and original extension
                    'parents': [training_images_folder_id]
                }
                copied_file = service.files().copy(fileId=image_id, body=file_metadata).execute()

                has_training_image = True
                break 

        except Exception as e:
            print(f"{image_name} threw an error: {e}")
            traceback.print_exc()  # This line prints the full traceback

    if not has_training_image:
        interns_without_training_data.append(folder['name'])



if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None
    st.session_state['download_zip_created'] = False
    st.session_state['creds'] = None
    # delete_collection('your-colleciton-id')
    st.session_state['begin_auth'] = False
    st.session_state['final_auth'] = False

st.title("Leadership Initiatives Photo Labeler")
st.caption("By Giacomo Pugliese")
with st.expander("Click to view full directions for this site"):
    st.subheader("User Credentials")
    st.write("- Either create or insert an existing program id to log into intern database (** a program ID is needed for all parts of the site **)")
    st.write("- Authenticate with Google before performing any actions (** Google authentication is needed for all parts of this site **)")
    st.subheader("Configure Training Data")
    st.write("- For the training data upload via google drive, upload a folder with all of the intern's folders, and ensure each folder has a least one solo image of them with the word 'bio' in its title (i.e. Giacomo Pugliese  - Bio - Middle School Jumpstart Program - July 9th .jpg).")
    st.write("- Alternatively, create student profiles and upload solo images of them to train the AI (** names must be in format FIRST_LAST **)")
    st.write("- Delete student profile to clear their training data if needed")
    st.subheader("Interns in System")
    st.write("- Displays all of the interns currently in your program's AI")
    st.subheader("Detect Interns in Photos")
    st.write("- Insert a comma seperated list of the folder links of your google drive containing intern photos")
    st.write("- Add a destination drive folder if you want the labeled intern folders to go somewhere different than the folder containing the input photos.")
    st.write("- Click 'Start Processing' and allow the AI to sort the images into individual student folders directly into the drive (may take a while, ensure that computer is on the whole time)")
    st.subheader("Renaming tool")
    st.write("- Insert the folder link of your google drive folder containing program's students at a particular location")
    st.write("- Choose the custom file ending for that location (i.e. ending would be '_Jumpstart_Group_1' for a file you want named 'Giacomo_Pugliese_2023_Jumpstart_Group_1')")
    st.write("- Click 'Start Renaming' and have the renamed pictures go right into the drive folder")


st.header('User Credentials')
st.subheader("Program Login")

col1, col2= st.columns(2)
deleted_collection = 0
display_programs = 0

with col1:
    collection_id = st.text_input("Enter your program ID to sign in", "")
    if collection_id == '':
        collection_id = 'your-default-collection-id'
    create_collection(collection_id)
    if st.button("View programs"):
        collections = list_collections()
        if collections:
            display_programs = 1
        else:
            display_programs = 2

with col2:
    deleted_program = st.text_input("Enter program ID to delete")
    if st.button("Delete this program") and deleted_program != 'your-default-collection-id':
            if(deleted_program not in list_collections()):
                st.error("Program ID doesn't exist")
            else:
                delete_collection(deleted_program)
                deleted_collection = 1

if(display_programs == 1):
    st.info(f"Current programs: {', '.join(collections)}")
    display_programs = 0
elif(display_programs == 2):
    st.info("No programs created yet!")

if(deleted_collection == 1):
    st.info("Program deleted from system")
    deleted_collection = 0



st.subheader("Google authentication")

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)
try:
    if st.button("Authenticate Google Account"):
        st.session_state['begin_auth'] = True
        # Request OAuth URL from the FastAPI backend
        response = requests.get(f"{'https://photo-labeler-842ac8d73e7a.herokuapp.com'}/auth?user_id={collection_id}")
        if response.status_code == 200:
            # Get the authorization URL from the response
            auth_url = response.json().get('authorization_url')
            st.markdown(f"""
                <a href="{auth_url}" target="_blank" style="color: #8cdaf2;">
                    Click to continue to authentication page


                </a>
                """, unsafe_allow_html=True)
            st.text("\n\n\n")
            # Redirect user to the OAuth URL
            # nav_to(auth_url)

    if st.session_state['begin_auth']:    
        if st.button("Finalize Google Authentication"):
            with st.spinner("Finalizing authentication..."):
                for i in range(6):
                    # Request token from the FastAPI backend
                    response = requests.get(f"{'https://photo-labeler-842ac8d73e7a.herokuapp.com'}/token/{collection_id}")
                    if response.status_code == 200:
                        st.session_state['creds'] = response.json().get('creds')
                        print(st.session_state['creds'])
                        st.success("Google account successfully authenticated.")
                        st.session_state['final_auth'] = True
                        break
                    time.sleep(1)
            if not st.session_state['final_auth']:
                st.error('Experiencing network issues, please refresh page and try again.')
                st.session_state['begin_auth'] = False
except:
    pass


# Add a person or image, or delete a person
st.header('Configure Training Data')


# Create a directory named after the collection
os.makedirs(collection_id, exist_ok=True)

st.subheader("Add training data using google drive folder")
# Drive directory link for bulk training data
training_data_directory_link = st.text_input("Enter a Google Drive directory link for bulk training data")

if st.button('Process Training Data'):
    if not st.session_state['final_auth']:
        st.error("Please authenticate with google!")
    elif collection_id == 'your-default-collection-id':
        st.error("Please enter a program id!")
    else:
        with st.spinner("Downloading training data..."):
            # Add a list to keep track of interns without training data
            interns_without_training_data = []
            
            # Google Drive service setup
            CLIENT_SECRET_FILE = 'credentials.json'
            API_NAME = 'drive'
            API_VERSION = 'v3'
            SCOPES = ['https://www.googleapis.com/auth/drive']

            with open(CLIENT_SECRET_FILE, 'r') as f:
                client_info = json.load(f)['web']

            creds_dict = st.session_state['creds']
            creds_dict['client_id'] = client_info['client_id']
            creds_dict['client_secret'] = client_info['client_secret']
            creds_dict['refresh_token'] = creds_dict.get('_refresh_token')

            # Create Credentials from creds_dict
            creds = Credentials.from_authorized_user_info(creds_dict)

            # Call the Drive v3 API
            service = build(API_NAME, API_VERSION, credentials=creds)
            # Extracting the folder ID from the link
            training_data_directory_id = training_data_directory_link.split('/')[-1]

            # Get all the sub-folders (interns' folders)
            query = f"'{training_data_directory_id}' in parents and trashed = false and mimeType = 'application/vnd.google-apps.folder'"
            intern_folders = service.files().list(q=query).execute().get('files', [])

            progress_report = st.empty()
            progress_report.text(f"Initializing training data...")
            i = 1
            # Check if 'Training Images' folder exists in the parent directory. If not, create it.
            query = f"'{training_data_directory_id}' in parents and name='Training Images' and trashed = false"
            results = service.files().list(q=query).execute().get('files', [])

            if results:
                training_images_folder_id = results[0]['id']
            else:
                file_metadata = {
                    'name': 'Training Images',
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [training_data_directory_id]
                }
                training_images_folder = service.files().create(body=file_metadata, fields='id').execute()
                training_images_folder_id = training_images_folder['id']
            with ProcessPoolExecutor(max_workers=15) as executor:
                futures = []
                for folder in intern_folders:
                    future = executor.submit(process_folder, folder, service, interns_without_training_data, collection_id, training_data_directory_id, )
                    futures.append(future)
                
                for future in as_completed(futures):
                    # If process_folder returns a result, handle it here
                    result = future.result()  # replace with appropriate handling if process_folder returns something
                    progress_report.text(f"Training progress: ({i}/{len(intern_folders)})")
                    i = i +1

        # After all interns have been processed, if there were interns without training data, display a Streamlit error
        if interns_without_training_data:
            st.error(f"The following interns have no properly formatted training data: {', '.join(interns_without_training_data)}")
        st.balloons()

if os.path.exists("converted3.jpg"):
    with open("converted3.jpg", "rb") as file:
            btn = st.download_button(
                label="Download converted3.jpg",
                data=file,
                file_name="converted3.jpg",
                mime="image/jpeg",
            )

st.subheader("Add training data manually")
person_name = st.text_input("Enter the intern's name")
person_image = st.file_uploader('Upload a solo image of the intern', type=['jpg', 'png'])

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button('Add this image'):
        if(collection_id == 'your-default-collection-id'):
            st.error("Please enter a program id!")
        else:
            if person_name and person_image and collection_id:
                # Save the image locally
                file_path = save_file_locally(person_image, person_name)
                # Upload the image to S3 bucket
                with open(file_path, "rb") as f:
                    upload_success = upload_file_to_s3(f, 'giacomo-aws-bucket', person_name)
                if upload_success:
                    # Check if person already exists
                    if person_name not in list_faces_in_collection(collection_id):
                        add_faces_to_collection('giacomo-aws-bucket', person_name, collection_id, person_name)
                        st.write('Person added successfully')
                    else:
                        # If person already exists, just add the image to the person's existing images in the collection
                        st.write('Image added to existing person')
                    # Clear the input fields
                    person_name = None
                    person_image = None
                else:
                    st.write('Failed to upload image')
            else:
                st.write('Please enter a name, upload an image, and provide a program id')

with col2:
    if st.button('Delete intern'):
        if(collection_id == 'your-default-collection-id'):
            st.error("Please enter a program id!")
        else:
            if person_name and collection_id:
                # Find the faceId of the person to delete
                face_id = next((face['FaceId'] for face in client.list_faces(CollectionId=collection_id)['Faces'] 
                                if face['ExternalImageId'] == person_name), None)
                if face_id:
                    delete_face_from_collection(collection_id, face_id)
                    st.write(f'Person {person_name} deleted successfully')
                else:
                    st.write(f'Person {person_name} not found')
            else:
                st.write('Please enter a name and program id to delete')


# Display the list of person names
person_names = list_faces_in_collection(collection_id)
st.header(f'Interns in System ({len(person_names)})')
st.write(', '.join(person_names))
if len(person_names) == 0:
    'No interns added yet.'
st.button("Refresh page")

########################################################################################
#    DETECT SECTION

def process_file_wrapper(args):
    return process_file(*args)

st.header('Detect Interns in Photos')
folder_links = st.text_area('Enter Google Drive Folder links (comma separated)')
destination_folder_link = st.text_input('Enter Google Drive Destination Folder link (Optional)')
start_processing = st.button('Start Processing')

if start_processing:
    if not folder_links:
        st.error("Please upload your google drive folders")
    elif collection_id == 'your-default-collection-id':
        st.error("Please enter a program id!")
    elif not st.session_state['final_auth']:
        st.error("Please authenticate with google!")
    else:
        folders = [x.strip() for x in folder_links.split(',')]
        match_dest = re.search(r'\/([a-zA-Z0-9-_]+)$', destination_folder_link) if destination_folder_link else None
        folder_ids = []
        error = False
        for folder_link in folders:
            match = re.search(r'\/([a-zA-Z0-9-_]+)$', folder_link)
            if(match is None):
                st.error(f'Invalid Google Drive link: {folder_link}. Please make sure the link is correct.')
                error = True
                break
            else:
                folder_id = match.group(1)
                folder_ids.append(folder_id)
        if not error:
            # If destination_folder_link is provided and valid, replace folder_id with destination_folder_id
            destination_folder_id = match_dest.group(1) if match_dest else folder_ids[0]

            CLIENT_SECRET_FILE = 'credentials.json'
            API_NAME = 'drive'
            API_VERSION = 'v3'
            SCOPES = ['https://www.googleapis.com/auth/drive']

            with open(CLIENT_SECRET_FILE, 'r') as f:
                client_info = json.load(f)['web']

            creds_dict = st.session_state['creds']
            creds_dict['client_id'] = client_info['client_id']
            creds_dict['client_secret'] = client_info['client_secret']
            creds_dict['refresh_token'] = creds_dict.get('_refresh_token')
            try:
                # Create Credentials from creds_dict
                creds = Credentials.from_authorized_user_info(creds_dict)

                # Call the Drive v3 API
                service = build(API_NAME, API_VERSION, credentials=creds)
            except:
                st.error("Please refresh the page and retry Google authentication.")

            # Create a dictionary to store each person's folder
            with st.spinner("Creating folders"):
                person_folder_dict = {}
                for person in person_names + ['Group Photos']:
                    folder_query = f"name='{person}' and '{destination_folder_id}' in parents and trashed=false"
                    folder_search = make_request_with_exponential_backoff(service.files().list(q=folder_query))
                    if not folder_search.get('files', []):
                        metadata = {'name': person, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [destination_folder_id]}
                        folder = make_request_with_exponential_backoff(service.files().create(body=metadata, fields='id'))
                    else:
                        folder = folder_search.get('files', [])[0]
                    print("hi!")
                    person_folder_dict[person] = folder

                person_images_dict = {}
                person_images_dict['Group Photos'] = []
                group_photo_threshold = 12

            progress_report = st.empty()

            with st.spinner("Labeling images.."):
                if not os.path.exists(f'{collection_id}/labels'):
                    os.makedirs(f'{collection_id}/labels')
                total_files = 0
                labeled_files = 0
                for folder_id in folder_ids:
                    page_token = None

                    # retrieve total amount of files
                    response = make_request_with_exponential_backoff(service.files().list(q=f"'{folder_id}' in parents and trashed=false and mimeType != 'application/vnd.google-apps.folder'",
                                                                                            spaces='drive',
                                                                                            fields='nextPageToken, files(id, name)',
                                                                                            pageToken=page_token,
                                                                                            pageSize=1000))
                    items = response.get('files', [])
                    total_files += len(items)

                progress_report.text(f"Labeling progress: ({0}/{total_files})")

                for folder_id in folder_ids:
                    page_token = None

                    while True:
                        response = make_request_with_exponential_backoff(service.files().list(q=f"'{folder_id}' in parents and trashed=false and mimeType != 'application/vnd.google-apps.folder'",
                                                                                            spaces='drive', 
                                                                                            fields='nextPageToken, files(id, name)',
                                                                                            pageToken=page_token,
                                                                                            pageSize=100))
                        items = response.get('files', [])
                        arguments = [(file, service, destination_folder_id, person_images_dict, group_photo_threshold, collection_id, person_folder_dict,) for file in items]

                        with ProcessPoolExecutor(max_workers=15) as executor:
                            futures = {executor.submit(process_file_wrapper, arg): arg for arg in arguments}
                            for future in as_completed(futures):
                                try:
                                    # Handling the future completion
                                    result = future.result()  # replace with appropriate handling if process_file_wrapper returns something
                                except:
                                    pass
                                labeled_files += 1
                                progress_report.text(f"Labeling progress: ({labeled_files}/{total_files})")

                        page_token = response.get('nextPageToken', None)
                        if page_token is None:
                            break

                consolidate_labels(collection_id)

                st.session_state['download_zip_created'] = True  
                st.balloons()

if 'download_zip_created' in st.session_state and st.session_state['download_zip_created']:  

    with open(f'{collection_id}/labels.txt', 'r') as f:
        st.download_button(
            label="Download all textual labels",
            data=f.read(),
            file_name='labels.txt',
            mime='text/plain'
        )


 
##############################################################################################

st.header('Naming Tool')
folder_id_rename = st.text_input('Enter Google Drive Folder ID for Renaming')
file_name_ending = st.text_input('Enter your custom file name ending')
start_renaming = st.button('Start Renaming')

if start_renaming and folder_id_rename:
    if(not folder_id_rename):
        st.error("Please upload your google drive folder")
    elif(collection_id == 'your-default-collection-id'):
        st.error("Please enter a program id!")
    elif not st.session_state['final_auth']:
        st.error("Please authenticate with google!")
    else:
        # Build the service
        creds = service_account.Credentials.from_service_account_file('credentials.json')
        service = build('drive', 'v3', credentials=creds)

        # Request files in the folder
        results = service.files().list(q=f"'{folder_id_rename}' in parents").execute()
        items = results.get('files', [])

        if not items:
            st.error("No files found.")
        else:
            total_files = len(items)

            progress_report = st.empty()  # Create a placeholder for the progress report

            for i, file in enumerate(items, start=1):
                try:
                    progress_report.text(f"Renaming progress: ({i}/{total_files})")  # Update the text in the placeholder

                    # Extract file extension
                    file_ext = os.path.splitext(file['name'])[1]

                    if '_2023_' in file['name']:
                        # Remove all characters past '_2023_' in the file name
                        new_file_name = re.sub(r'_2023_.*', '', file['name'])

                        # Append the custom file name ending and the file extension
                        new_file_name += f'_2023_{file_name_ending}{file_ext}'

                        # Rename the file
                        service.files().update(fileId=file['id'], body={"name": new_file_name}).execute()

                except Exception as e:
                    st.write(f"Error renaming {file['name']}: {e}")
                    continue

            st.success("All files renamed successfully!")

