FROM astrocrpublic.azurecr.io/runtime:3.1-14

USER root

# Install unixODBC, GnuPG, etc. and MS ODBC DRIVER 17
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg2 \
        libgssapi-krb5-2 \
        unixodbc \
        unixodbc-dev && \
    # Import Microsoft GPG key in the Debian 12 way
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | \
        gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    # Add the Debian 12 MS SQL / ODBC repo
    curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list | \
        tee /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    # Install ODBC DRIVER 17 for SQL Server
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 && \
    rm -rf /var/lib/apt/lists/*

# Back to astro user
USER astro

# === Install Python dependencies from requirements.txt ===
WORKDIR /usr/local/airflow

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
# ========================================================
EXPOSE 8501
