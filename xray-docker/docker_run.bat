docker run -d --name xray-detector ^
  -p 5000:5000 ^
  -v C:\models\F_M1_90:/app/Models/F_M1_90 ^
  -v %cd%\uploads:/app/uploads ^
  xray-docker:latest