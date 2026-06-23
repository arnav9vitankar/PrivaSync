# FROM baseImage

FROM node:22-alpine 
# gives node.js

WORKDIR /app
# working directory app

COPY package*.json ./
# copies dependencies(package.json and package-lock.json)

RUN npm ci
# ci(clean install)

COPY . .

RUN npm run build
# compiles,optimizes and bundles our source code into production ready package

EXPOSE 3000
# Next.js runs on 3000

CMD ["npm", "start"]