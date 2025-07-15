docker run --name web-app-for-pas ^
  -p 5000:5000 ^
  -v D:\GitHub\PAS-models\F_M1_90:/app/models/F_M1_90 ^
  -v %cd%\uploads:/app/uploads ^
  xray-docker:latest
pause