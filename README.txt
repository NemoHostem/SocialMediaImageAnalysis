The engagement Ground Truth as well as the list of the images (Flickr IDs) are stored in the directory 'Engagement'.
Each file si a data pickle that contains the sequences for three different groups (10, 20 or 30 days) and three kind of engagement scores (views, comments, favorites).

The data are split into two sets: Dataset A (DSA) and Dataset B (DSB).
Each group containts roughtly 10K images.
The data are stored in sqlite3 databases with the following table structure:

CREATE TABLE headers(Id INTEGER PRIMARY KEY, 
			FlickrId TEXT, -- Id of the image on Flickr
			Day INT, 
			UserId TEXT, -- Id of the user on Flickr
			URL TEXT, -- URL of the image on Flickr
			Path TEXT, 
			DatePosted TEXT, -- Timestamp of the date of the image post on Flickr
			DateTaken, -- Timestamp of the claimed date of the image creation
			DateCrawl TEXT) -- Timestamp of the crawling time

CREATE TABLE image_info(Id INTEGER PRIMARY KEY, -
			FlickrId TEXT, -- Id of the image on Flickr
			Day INT, 
			Camera TEXT, 
			Size INT, -- Total number of pixel of the original image
			Title TEXT, -- Title of the post
			Description TEXT, -- Description of the post
			NumSets INT, -- Number of albums the photo is shared in
			NumGroups INT, -- Number of groups the photo is shared in
			AvgGroupsMemb REAL, -- Avg number of members of the groups in which the photo is shared
			AvgGroupPhotos REAL, -- Avg number of photos of the groups in which the photo is shared
			Tags TEXT, -- Social tags of the post
			Latitude TEXT, 
			Longitude TEXT, 
			Country TEXT)

CREATE TABLE user_info(Id INTEGER PRIMARY KEY, 
			UserId TEXT, -- Id of the user on Flickr
			Day INT, 
			Username TEXT, 
			Ispro INT, 
			HasStats INT, 
			Contacts INT, -- Number of contacts of the user on Flickr
			PhotoCount INT, -- Number of photos of the user
			MeanViews REAL, -- Mean number of views of the user's photos
			GroupsCount INT, -- Number of groups the user is enrolled in
			GroupsAvgMembers REAL, -- Avg number of members of the groups in which the the user is enrolled
			GroupsAvgPictures REAL) -- Avg number of photos of the groups in which the the user is enrolled
