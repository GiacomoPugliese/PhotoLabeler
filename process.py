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
from urllib.parse import urlparse, parse_qs
# import pyheif

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

# Define function to add faces to the collection
def add_faces_to_collection(bucket, photo, collection_id, external_image_id):
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
    response = client.index_faces(
        CollectionId=collection_id,
        Image={'S3Object':{'Bucket':bucket,'Name':photo}},
        ExternalImageId=external_image_id,
        MaxFaces=5,
        QualityFilter="AUTO",
        DetectionAttributes=['ALL']
    )

    return response['FaceRecords']

# Define function to list faces in the collection
def list_faces_in_collection(collection_id):
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
    response = client.list_faces(CollectionId=collection_id)
    return [face['ExternalImageId'] for face in response['Faces']]

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

def find_matching_faces(photo, collection_id):
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

def resize_image(image, basewidth):
    img = Image.open(image)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img

def process_file_wrapper(args):
    return process_file(*args)

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

def process_file(file, service, folder_id, person_images_dict, group_photo_threshold, collection_id, person_folder_dict):
    print(f"{file['name']} started")
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
            try:
                heif_file = pyheif.read(fh.getvalue())
                img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode)
                byte_arr = io.BytesIO()
                img.save(byte_arr, format='JPEG')
                img.save('converted3.jpg')
                byte_img = byte_arr.getvalue()
            except:
                img_io = io.BytesIO(fh.getvalue())
                img = resize_image(img_io, 1000)
                if img.mode != 'RGB':  # Convert to RGB if not already
                    img = img.convert('RGB')
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
            # Check if file already exists in the destination folder
            search_response = make_request_with_exponential_backoff(service.files().list(q=f"name='{new_file_name}' and '{folder['id']}' in parents and trashed=false",
                                                                                        spaces='drive',
                                                                                        fields='files(id, name)'))

            # If file does not exist, then copy it
            if not search_response.get('files', []):
                copied_file = make_request_with_exponential_backoff(service.files().copy(fileId=file['id'], body={"name": new_file_name, "parents": [folder['id']]}))


    except Exception as e:
        print(f"{file['name']} threw an error: {e}")
        traceback.print_exc()  # This line prints the full traceback

    # Generate a unique filename using uuid library
    unique_filename = str(uuid.uuid4()) + '.txt'
    with open(f'{collection_id}/labels/{unique_filename}', 'w') as f:
        # Write the image name and persons detected to the file
        f.write(f"{file['name']}: {', '.join(set(persons))}")

    print(f"{file['name']}: {', '.join(set(persons))}")
    return file['name']

def process_files(folder_ids, service, destination_folder_id, person_images_dict, group_photo_threshold, collection_id, person_folder_dict, labeled_files, total_files, cache, progress_report):
    flag = False
    for folder_id in folder_ids:
        try:
            page_token = None

            while True:
                response = make_request_with_exponential_backoff(service.files().list(q=f"'{folder_id}' in parents and trashed=false and mimeType != 'application/vnd.google-apps.folder'",
                                                                                    spaces='drive', 
                                                                                    fields='nextPageToken, files(id, name)',
                                                                                    pageToken=page_token,
                                                                                    pageSize=1000))
                items = response.get('files', [])
                new_items = [file for file in items if file['name'] not in cache['labeled_files']]
                items = new_items

                arguments = [(file, service, destination_folder_id, person_images_dict, group_photo_threshold, collection_id, person_folder_dict,) for file in items]
                with ProcessPoolExecutor(max_workers=15) as executor:
                    futures = {executor.submit(process_file_wrapper, arg): arg for arg in arguments}
                    for future in as_completed(futures):
                        try:
                            result = future.result()  # replace with appropriate handling if process_file_wrapper returns something
                            cache['labeled_files'].append(result)
                            labeled_files += 1
                            cache['file_progress'] += 1
                            remaining_time = (total_files - max(labeled_files, cache['file_progress'])) * (1/30)
                            flag = True
                            # progress_report.text(f"Labeling progress: {max(labeled_files, cache['file_progress'])}/{total_files} ({round(remaining_time, 1)} minutes remaining)")
                        except Exception as e:
                            labeled_files += 1
                            cache['file_progress'] += 1
                            remaining_time = (total_files - max(labeled_files, cache['file_progress'])) * (1/30)
                            flag = True
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
        except:
            pass

    consolidate_labels(collection_id)
    if not flag:
        # progress_report.text("")
        # progress_report_folder.text("")
        pass

    return cache