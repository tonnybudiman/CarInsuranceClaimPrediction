mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"t.f.budiman@gmail.com\"\n\
" > ~/.streamllit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml