# Sample Queries for Ethereum BigQuery Dataset

This document collects sample SQL queries from three public tutorials about the BigQuery Ethereum dataset. These queries are reproduced verbatim and can be used to understand the schema, compute wallet balances and build recommendation systems. Each query includes a brief description, use them as reference to build your own queries.

Consider when using the column hash, it happens to be a keyword in BigQuery, so to indicate that hash is the name of the column, we enclose it in backticks. For example:
```sql
WHERE `hash` =
```

## Query to discover the available tables within this Ethereum dataset

```sql
SELECT *
FROM `bigquery-public-data.crypto_ethereum.INFORMATION_SCHEMA.TABLES`;
```

## Get balance of the address 0xcda7559bcef42e68f16233b5b8c99c757a5f4697

```sql
SELECT address, eth_balance
FROM `bigquery-public-data.crypto_ethereum.balances`
WHERE address = '0xcda7559bcef42e68f16233b5b8c99c757a5f4697';
```

## Get Top 10 Addresses with Highest Balances

```sql
SELECT address, eth_balance
FROM `bigquery-public-data.crypto_ethereum.balances`
ORDER BY eth_balance DESC
LIMIT 10;
```

## Get transactions by transaction hash 0x1f5abf832162265242a27b7f8bf6a6ec6b2f4a4fd6d9e681cab6924807636fd7

```sql
SELECT *
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE `hash` = "0x1f5abf832162265242a27b7f8bf6a6ec6b2f4a4fd6d9e681cab6924807636fd7"
	AND block_timestamp > TIMESTAMP '2023-09-06 00:00:00';
```

## Get latest transaction from an address 0xcda7559bcef42e68f16233b5b8c99c757a5f4697

```sql
SELECT *
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE (from_address = '0xcda7559bcef42e68f16233b5b8c99c757a5f4697'
	OR to_address = '0xcda7559bcef42e68f16233b5b8c99c757a5f4697')
ORDER BY block_timestamp DESC
LIMIT 1;
```

## Get last 10 token transfers of address 0xcda7559bcef42e68f16233b5b8c99c757a5f4697

```sql
SELECT *
FROM `bigquery-public-data.crypto_ethereum.token_transfers`
WHERE (from_address = '0xcda7559bcef42e68f16233b5b8c99c757a5f4697'
	OR to_address = '0xcda7559bcef42e68f16233b5b8c99c757a5f4697')
ORDER BY block_timestamp DESC
LIMIT 10;
```

## Get Latest 10 USDT Transfers 0xdAC17F958D2ee523a2206206994597C13D831ec7

```sql
SELECT *
FROM `bigquery-public-data.crypto_ethereum.token_transfers`
WHERE
	token_address = ‘0xdAC17F958D2ee523a2206206994597C13D831ec7’
	AND block_timestamp > TIMESTAMP '2023-09-04 00:00:00'
ORDER BY block_timestamp DESC
LIMIT 10;
```

## Get token ratings. This SQL queries top 1000 tokens by transfers count, calculates the balances for each token, and outputs (token_address, user_address, rating) triples. Rating there is calculated as the percentage of supply held by the user. This filter — where balance/supply * 100 > 0.001 — prevents airdrops appearing in the result.

```sql
with top_tokens as (
  select token_address, count(1) as transfer_count
  from `bigquery-public-data.crypto_ethereum.token_transfers` as token_transfers
  group by token_address
  order by transfer_count desc
  limit 1000
),
token_balances as (
    with double_entry_book as (
        select token_address, to_address as address, cast(value as float64) as value, block_timestamp
        from `bigquery-public-data.crypto_ethereum.token_transfers`
        union all
        select token_address, from_address as address, -cast(value as float64) as value, block_timestamp
        from `bigquery-public-data.crypto_ethereum.token_transfers`
    )
    select double_entry_book.token_address, address, sum(value) as balance
    from double_entry_book
    join top_tokens on top_tokens.token_address = double_entry_book.token_address
    where address != '0x0000000000000000000000000000000000000000'
    group by token_address, address
    having balance > 0
),
token_supplies as (
    select token_address, sum(balance) as supply
    from token_balances
    group by token_address
)
select 
    token_balances.token_address, 
    token_balances.address as user_address, 
    balance/supply * 100 as rating
from token_balances
join token_supplies on token_supplies.token_address = token_balances.token_address
where balance/supply * 100 > 0.001
```

## Get live balance of the top 10 addresses. Considering that the balances table is only a snapshot. 

```sql
#standardSQL
-- MIT License
-- Copyright (c) 2018 Evgeny Medvedev, evge.medvedev@gmail.com
with double_entry_book as (
    -- debits
    select to_address as address, value as value
    from `bigquery-public-data.crypto_ethereum.traces`
    where to_address is not null
    and status = 1
    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)
    union all
    -- credits
    select from_address as address, -value as value
    from `bigquery-public-data.crypto_ethereum.traces`
    where from_address is not null
    and status = 1
    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)
    union all
    -- transaction fees debits
    select miner as address, sum(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value
    from `bigquery-public-data.crypto_ethereum.transactions` as transactions
    join `bigquery-public-data.crypto_ethereum.blocks` as blocks on blocks.number = transactions.block_number
    group by blocks.miner
    union all
    -- transaction fees credits
    select from_address as address, -(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value
    from `bigquery-public-data.crypto_ethereum.transactions`
)
select address, sum(value) as balance
from double_entry_book
group by address
order by balance desc
limit 10
```

## Sources:

* https://bitquery.io/blog/querying-bigquery-blockchain-dataset
* https://medium.com/google-cloud/building-token-recommender-in-google-cloud-platform-1be5a54698eb
* https://medium.com/google-cloud/how-to-query-balances-for-all-ethereum-addresses-in-bigquery-fb594e4034a7

