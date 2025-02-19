# Use an official Node.js runtime as a parent image
FROM node:20-alpine AS build

# Inject the backend URL at build time
# ARG VITE_BACKEND_URL=http://backend:8000
# #ARG VITE_BACKEND_URL=http://courserecommendersite.courserecommendersite.courserecommendersite.local:8000
# ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

# Set the working directory to /app
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the app
RUN npm run build

# Stage 2 - Serve the app with Nginx
FROM nginx:stable-alpine

# Copy the dist directory (Vite's build output) to Nginx's html directory
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start Nginx when the container launches
CMD ["nginx", "-g", "daemon off;"]