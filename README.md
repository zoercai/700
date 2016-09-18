# Clusterr
UoA SOFTENG 700 project #72: Grouping similar newspaper articles

## Description:
Grouping related documents involves entity recognition and then measuring how similar the meaning of the entities are. Both these tasks are complex. In this project, you'll tackle the second task. You will manually identify the entities of interest and build a system that does probabilistic data matching of newspaper articles based on the identified entities. The results of the matching will be visualized. The technique will be compared against state of the art algorithms.

## Outcome:
A system that groups related newspaper articles

## Supervisor
Gill Dobbie

## Team
Ruoyi (Zoe) Cai
Chanjun Park

## Github
https://github.com/zoercai/Clusterr

---

# How to Run the Application

## Locally (Allows more than 30 articles to be retrieved)
* Get python3: https://www.python.org/downloads/

* Install virtualenv by running the command:
`[sudo] pip3 install virtualenv`
(For more information on virutalenv installation, visit: https://virtualenv.pypa.io/en/stable/installation/)

* Navigate to the project folder: `path/to/Clusterr/`

* Create a virtualenv by running the command:
`virtualenv venv`
This creates a folder named venv in the project directory.

* Activate and enter the virtualenv by running the command:
`. venv/bin/activate`

* Install all project dependencies:
`pip3 install -r requirements.txt`

* Once all dependencies are installed, start the web application:
`python3 Application/MainRunner.py`

* Then go to `http://127.0.0.1:5000/` (or whatever address is displayed) on your web browser.


## Online (A maximum of 30-40 articles can be retrieved due to free server restrictions)
Visit http://clusterr.zoecai.com
