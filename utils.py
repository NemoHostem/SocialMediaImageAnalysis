import sqlite3 as lite
import pickle as pkl
import numpy as np
import json
import os
import sys

dbs_A_path = os.path.join(sys.path[0], '/SIPD18/DSA')
dbs_B_path = os.path.join(sys.path[0], '/SIPD18/DSB')


def get_user_info(db,us_id):
    con = lite.connect(db)
    cur = con.cursor()
    cur.execute('SELECT Ispro, Contacts, PhotoCount, MeanViews, GroupsCount, GroupsAvgMembers, GroupsAvgPictures FROM user_info WHERE UserId = \''+us_id+'\'')
    res = cur.fetchall()
    res = res[0]
    return np.array(res)

def get_image_info(db,img_id):
    con = lite.connect(db)
    cur = con.cursor()
    cur.execute('SELECT Size, Title, Description, NumSets, NumGroups, AvgGroupsMemb, AvgGroupPhotos, Tags FROM image_info WHERE FlickrId = \''+img_id+'\'')
    res = cur.fetchall()
    res = res[0]

    feat = []    

    feat.append(res[0])                         # Image size
    feat.append(len(res[1]))                         # Title length
    feat.append(len(res[2]))                         # Description length
    feat.append(res[3])    
    feat.append(res[4])
    feat.append(res[5])
    feat.append(res[6])
    tags = json.loads(res[7])                # res[7] is a json list of tags
    feat.append(len(tags))                         #number of tags

    return feat, np.array(res)


def load_social_features(ids_list, path_A, path_B):

    feat_data = []

    con = lite.connect(path_A+'/headers.db')
    cur = con.cursor()
    cur.execute('SELECT FlickrId, UserId FROM headers')
    HD_A = cur.fetchall()
    con.close()
    DS_A = [x[0] for x in HD_A]

    con = lite.connect(path_B+'/headers.db')
    cur = con.cursor()
    cur.execute('SELECT FlickrId, UserId FROM headers')
    HD_B = cur.fetchall()
    con.close()
    DS_B = [x[0] for x in HD_B]

    print("\nLoading features:")

    HD = []
    for flickr_id in ids_list:
        print('\n')
        try:
            im_idx = DS_A.index(flickr_id)        
            print("FlickrId:\t"+flickr_id+"\tin dataset A.")
            HD = HD_A
            db_path = path_A
        except ValueError:
            im_idx = DS_B.index(flickr_id)        
            print("FlickrId:\t"+flickr_id+"\tin dataset B.")
            HD = HD_B
            db_path = path_B

        user_id = HD[im_idx][1]
        print("Getting user info...")
        user_feat = get_user_info(db_path+'/user_info.db',user_id)
    
        print("Getting image info...")
        image_feat, _ = get_image_info(db_path+'/image_info.db',flickr_id)
            
        feat_data.append(list(user_feat) + list(image_feat))
    return feat_data

# The data loaded from the .pickle file is a list of elements with the following structure:
#
#   [ [photo_id, time, engagement],
#   [photo_id, time, engagement],
#   [photo_id, time, engagement],
#   ...
#   [photo_id, time, engagement] ]
#
#   photo_id:     Flickr Id of the photo
#   time:         timestamp related to the engagement score
#   engagement:   engagement score (i.e., views, comments or favorites) at timestamp 'time'

with open(os.path.join(sys.path[0], '/Engagement/views_30_days.pickle'),'rb') as f:
    data = pkl.load(f)
    flickr_ids = [x[0] for x in data]
    sequences = [x[2] for x in data]
    

X = load_social_features(flickr_ids,dbs_A_path,dbs_B_path)
