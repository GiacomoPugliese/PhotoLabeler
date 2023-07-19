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
        st.write('Deleted collection:', collection_id)
    except ClientError as e:
        logging.error(e)
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
                FaceMatchThreshold=70,
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
    name = re.sub(r' -.*|_.*|.jpg', '', name)
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

    # Write the consolidated labels to the final text file
    with open(f'{collection_id}/labels.txt', 'w') as f:
        for person, images in labels_dict.items():
            f.write(f'{person}: {", ".join(images)}\n\n')

    st.session_state['download_zip_created'] = True

if 'person_names' not in st.session_state:
    st.session_state['person_names'] = []
    st.session_state['last_uploaded_file'] = None
    st.session_state['download_zip_created'] = False
    st.session_state['creds'] = None
    # delete_collection('your-colleciton-id')
    st.session_state['begin_auth'] = False
    st.session_state['final_auth'] = False

st.title("Leadership Initiatives Photo Labeler")

with st.expander("Click to view full directions for this site"):
    st.subheader("Configure Training Data")
    st.write("- Either create or sign in to your program by entering a program ID (** a program ID is needed for all parts of the site **)")
    st.write("- Create student profiles and upload solo images of them to train the AI (** names must be in format FIRST_LAST **)")
    st.write("- Delete student profile to clear their training data if needed")
    st.subheader("Interns in System")
    st.write("- Displays all of the interns currently in your program's AI")
    st.subheader("Detect Interns in Photos")
    st.write("- Insert the folder link of your google drive containing your photos")
    st.write("- Share the folder with giacomo@photolabeler-393105.iam.gserviceaccount.com")
    st.write("- Click 'Start Processing' and allow the AI to sort the images into individual student folders directly into the drive (may take a while)")
    st.subheader("Renaming tool")
    st.write("- Insert the folder links of your google drive containing program's students at a particular location")
    st.write("- Choose the custom file ending for that location (i.e. ending would be '_Jumpstart_Group_1' for a file you want named 'Joe_Random_2023_Jumpstart_Group_1')")
    st.write("- Click 'Start Renaming' and download a zip folder of the automatically renamed image files")


st.header('User Credentials')
collection_id = st.text_input("Enter your program ID", "")
if collection_id == '':
    collection_id = 'your-default-collection-id'
# collection_id = 'your-collection-id'
create_collection(collection_id)

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
                for i in range(5):
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
                st.error('Failed to retrieve credentials. Please refresh page and try again.')
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
    i = 1
    for folder in intern_folders:
        # Get all the images in the intern's folder
        query = f"'{folder['id']}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
        intern_images = service.files().list(q=query).execute().get('files', [])
        progress_report.text(f"Labeling progress: ({i}/{len(intern_folders)})")
        i = i +1
        # Iterate over intern's images and find one with 'bio' in the file name
        for img in intern_images:
            try:
                if 'bio' in img['name'].lower():  # Case-insensitive search
                    # Get the image
                    image_id = img['id']
                    image_name = img['name']
                    request = service.files().get_media(fileId=image_id)
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        _, done = downloader.next_chunk()

                    # Save the image to a temporary local file
                    temp_file_path = 'tempo_image.jpg'
                    with open(temp_file_path, 'wb') as f:
                        f.write(fh.getvalue())
                        
                    # Process image based on its type
                    if img['name'].endswith('.heic') or img['name'].endswith('.HEIC'):
                        heif_file = pyheif.read(temp_file_path)
                        img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode)
                    else:
                        img = Image.open(temp_file_path)
                        
                    # Correct orientation and resize image
                    img = correct_image_orientation(img)
                    img = resize_image(temp_file_path, 1000)

                    # If image is not RGB, convert it to RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Save the corrected and resized image to a BytesIO object
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format='JPEG')
                    byte_img = byte_arr.getvalue()

                    # Remember to remove the temporary file if you don't need it anymore
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

                    
                    # Save the corrected and resized image locally
                    with open('training_img.jpg', 'wb') as out:
                        out.write(byte_img)
                    # Get the intern's name from the folder name
                    intern_name = folder['name']
                    sanitized_intern_name = sanitize_name(intern_name)
                    # Check if person already exists
                    if sanitized_intern_name not in list_faces_in_collection(collection_id):
                        # If not, add the intern to the collection with the training image
                        with open('training_img.jpg', 'rb') as img_file:
                            upload_success = upload_file_to_s3(img_file, 'giacomo-aws-bucket', sanitized_intern_name)
                            if upload_success:
                                add_faces_to_collection('giacomo-aws-bucket', sanitized_intern_name, collection_id, sanitized_intern_name)
                                st.session_state['person_names'].append(sanitized_intern_name)
                                print(f'Person {sanitized_intern_name} added successfully')
                            else:
                                print('Failed to upload image')
                    else:
                        # If the person already exists, add the image to the person's existing images in the collection
                        add_training_image_to_person(collection_id, sanitized_intern_name, 'training_img.jpg')
                        print(f'Image added to existing person {sanitized_intern_name}')
                    break  # Exit the loop as we've found a suitable image
            except Exception as e:
                print(f"{image_name} threw an error: {e}")

st.subheader("Add training data manually")
person_name = st.text_input("Enter the intern's name")
person_image = st.file_uploader('Upload a solo image of the intern', type=['jpg', 'png'])

if st.button('Add image'):
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
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
                    st.session_state['person_names'].append(person_name)
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
            st.write('Please enter a name, upload an image, and provide a collection ID')




if st.button('Delete intern'):
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
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
            st.write('Please enter a name and collection ID to delete')

# Display the list of person names
person_names = list_faces_in_collection(collection_id)
st.header(f'Interns in System ({len(person_names)})')
st.write(', '.join(person_names))
if len(person_names) == 0:
    'No interns added yet.'
st.button("Refresh page")

########################################################################################
#    DETECT SECTION

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
            byte_img = byte_arr.getvalue()
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

def process_file_wrapper(args):
    return process_file(*args)

st.header('Detect Interns in Photos')
folder_links = st.text_input('Enter Google Drive Folder links (comma separated)')
destination_folder_link = st.text_input('Enter Google Drive Destination Folder link (Optional)')
start_processing = st.button('Start Processing')

if start_processing:
    if not folder_links:
        st.error("Please upload your google drive folders")
    else:
        folders = [x.strip() for x in folder_links.split(',')]
        match_dest = re.search(r'\/([a-zA-Z0-9-_]+)$', destination_folder_link) if destination_folder_link else None
        folder_ids = []
        for folder_link in folders:
            match = re.search(r'\/([a-zA-Z0-9-_]+)$', folder_link)
            if(collection_id == 'your-default-collection-id' or match is None):
                if match is None:
                    st.error(f'Invalid Google Drive link: {folder_link}. Please make sure the link is correct.')
                else:
                    st.error("Please enter a collection id!")
            else:
                folder_id = match.group(1)
                folder_ids.append(folder_id)

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
            for person in person_names:
                folder_query = f"name='{person}' and '{destination_folder_id}' in parents and trashed=false"
                folder_search = make_request_with_exponential_backoff(service.files().list(q=folder_query))
                if not folder_search.get('files', []):
                    metadata = {'name': person, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [destination_folder_id]}
                    folder = make_request_with_exponential_backoff(service.files().create(body=metadata, fields='id'))
                else:
                    folder = folder_search.get('files', [])[0]

                person_folder_dict[person] = folder

            person_images_dict = {}
            person_images_dict['Group Photos'] = []
            group_photo_threshold = 15

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
                            # Handling the future completion
                            result = future.result()  # replace with appropriate handling if process_file_wrapper returns something
                            labeled_files += 1
                            progress_report.text(f"Labeling progress: ({labeled_files}/{total_files})")

                    page_token = response.get('nextPageToken', None)
                    if page_token is None:
                        break

            consolidate_labels(collection_id)

            st.session_state['download_zip_created'] = True  

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
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
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

