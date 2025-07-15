docker run --name web-app-for-pas ^
  -p 5000:5000 ^
  -v "D:/GitHub/PAS-models/F_M1_90":/models/F_M1_90 ^
  -v "D:/GitHub/Web-App-for-PAS/xray-docker/uploads":/uploads ^
  xray-docker:latest
pause