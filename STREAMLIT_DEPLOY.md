# 🚀 Deploy to Streamlit Cloud - Commands

## Step 1: Push Code to GitHub

```bash
# Check git status
git status

# Add all files
git add .

# Commit changes
git commit -m "Deploy to Streamlit Cloud"

# If you haven't set up remote yet:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

## Step 2: Deploy on Streamlit Cloud

1. Go to: **https://share.streamlit.io**
2. Sign in with your **GitHub** account
3. Click **"New app"**
4. Fill in:
   - **Repository**: Select your repository
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: (auto-generated or custom)
5. Click **"Deploy"**

## Step 3: Your App URL

Your app will be live at:
```
https://your-app-name.streamlit.app
```

## Update Deployment

```bash
git add .
git commit -m "Update app"
git push origin main
# Streamlit Cloud auto-redeploys!
```

