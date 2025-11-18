# Deployment Guide - Customer Churn Prediction App

## 🚀 Deploying to Render

This guide will walk you through deploying the Customer Churn Prediction Streamlit app to Render.

## Prerequisites

- GitHub account
- Render account ([sign up here](https://render.com))
- Your code pushed to a GitHub repository

## Method 1: Using render.yaml (Recommended)

### Step 1: Push Code to GitHub

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Customer Churn Prediction App"

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

### Step 2: Deploy on Render

1. **Go to Render Dashboard**
   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Sign in or create an account

2. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub account if not already connected
   - Select your repository (`ml-dravit` or your repo name)

3. **Render Auto-Detection**
   - Render will automatically detect `render.yaml`
   - Configuration will be pre-filled
   - Review settings:
     - **Name**: `customer-churn-prediction`
     - **Environment**: Python 3
     - **Plan**: Free (or upgrade for better performance)

4. **Deploy**
   - Click "Create Web Service"
   - Render will:
     - Clone your repository
     - Install dependencies from `requirements.txt`
     - Start the Streamlit app
   - Wait 2-5 minutes for deployment

5. **Access Your App**
   - Once deployed, your app will be live at:
   - `https://customer-churn-prediction.onrender.com`
   - (or your custom domain if configured)

## Method 2: Manual Configuration

If you prefer manual setup:

1. **Create New Web Service** on Render
2. **Configure Settings**:
   - **Name**: `customer-churn-prediction`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: 
     ```
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
   - **Plan**: Free tier

3. **Environment Variables** (Optional):
   - `PYTHON_VERSION`: `3.11.0`
   - `KAGGLE_USERNAME`: (if using Kaggle API)
   - `KAGGLE_KEY`: (if using Kaggle API)

4. **Deploy**

## Environment Variables

### Optional: Kaggle API Credentials

If you want to use Kaggle dataset download feature:

1. Get your Kaggle API credentials:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Scroll to "API" section
   - Click "Create New Token"
   - Download `kaggle.json`

2. Add to Render Environment Variables:
   - `KAGGLE_USERNAME`: Your Kaggle username
   - `KAGGLE_KEY`: Your Kaggle API key

**Note**: The app also supports file upload, so Kaggle credentials are optional.

## Troubleshooting

### Build Fails

**Issue**: Build command fails
- **Solution**: Check `requirements.txt` has all dependencies
- Verify Python version compatibility (3.8+)

**Issue**: Import errors
- **Solution**: Ensure all packages are listed in `requirements.txt`
- Check for version conflicts

### App Doesn't Start

**Issue**: App crashes on startup
- **Solution**: 
  - Check Render logs for error messages
  - Verify `app.py` is in root directory
  - Ensure start command is correct

**Issue**: Port binding error
- **Solution**: Start command should use `$PORT` environment variable
- Verify `--server.address 0.0.0.0` is included

### Performance Issues

**Issue**: Slow loading
- **Solution**: 
  - Free tier apps spin down after 15 min inactivity
  - First request after spin-down takes 30-60 seconds
  - Consider upgrading to paid tier for always-on

**Issue**: Memory errors
- **Solution**: 
  - Reduce dataset size for testing
  - Optimize model training (reduce CV folds)
  - Upgrade to paid tier for more resources

### Kaggle Download Issues

**Issue**: Kaggle dataset download fails
- **Solution**:
  - Add Kaggle credentials as environment variables
  - Or use "Upload File" option in the app
  - Check internet connectivity in Render logs

## Monitoring

### View Logs

1. Go to your service in Render dashboard
2. Click "Logs" tab
3. View real-time logs and errors

### Health Checks

Render automatically monitors your app:
- Checks if app responds on the configured port
- Restarts if app crashes
- Sends email notifications for failures

## Updating Your App

### Automatic Deploys

Render automatically redeploys when you push to your main branch:
```bash
git add .
git commit -m "Update app"
git push origin main
```

### Manual Deploys

1. Go to Render dashboard
2. Click "Manual Deploy" → "Deploy latest commit"

## Custom Domain (Optional)

1. Go to your service settings
2. Click "Custom Domains"
3. Add your domain
4. Follow DNS configuration instructions

## Cost Considerations

### Free Tier
- ✅ Free forever
- ⚠️ Apps spin down after 15 min inactivity
- ⚠️ Limited resources (512MB RAM)
- ⚠️ Slower cold starts

### Paid Tier ($7/month)
- ✅ Always-on apps
- ✅ More resources (512MB RAM)
- ✅ Faster performance
- ✅ Better for production use

## Security Best Practices

1. **Don't commit secrets**: Use environment variables
2. **Enable HTTPS**: Automatically enabled on Render
3. **Rate limiting**: Consider adding for production
4. **Input validation**: Already implemented in the app

## Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Render Support**: [render.com/support](https://render.com/support)
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)

---

**Happy Deploying! 🎉**

