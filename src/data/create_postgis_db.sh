# Create a postgis container based on [postgis/docker-postgis: Docker image for PostGIS](https://github.com/postgis/docker-postgis)
# We use this to ingest test-data, in the order od ~30MB
# The complete dataset is several hundreds GB, we suggest store the data on your machine binding volumes like:
# docker run -d \
#     --name some-postgres \
#     -e POSTGRES_PASSWORD=mysecretpassword \
#     -e PGDATA=/var/lib/postgresql/data/pgdata \
#     -v /tmp/postgres-data:/var/lib/postgresql/data \
#     postgis/postgis
docker run -d \
    --name test-postgis \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=mysecretpassword \
    postgis/postgis
#########################################################
# connect from your system with:
#
# psql -h localhost -p 5432  -U postgres
#
# or from within the container :
#
# docker exec -ti test-postgis psql -U postgre
