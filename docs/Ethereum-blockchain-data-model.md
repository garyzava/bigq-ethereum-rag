# Ethereum blockchain (BigQuery) data model

**Dataset description**

* **Source & update cadence** – The Ethereum ETL project extracts raw blocks from Ethereum and publishes the data as a public dataset in BigQuery. The Kaggle dataset is a mirror of this public dataset. The documentation explains that all tables are published in BigQuery so that users **don’t need to export the chain themselves**, and the data is updated near real‑time with only a \~4‑minute delay to allow for block finality[\[1\]](https://ethereum-etl.readthedocs.io/en/latest/google-bigquery/#:~:text=Querying%20in%20BigQuery). This makes the dataset suitable for historical or near‑current analytics on the Ethereum mainnet.  
* **Scope of data** – The dataset contains the entire Ethereum blockchain from the genesis block to the present (updated near real‑time). It includes tables covering blocks, transactions, receipts, logs, traces, contracts, tokens, token transfers, balances and supplementary tables such as sessions, load\_metadata and amended\_tokens. The Ethereum ETL repository describes the ETL jobs used to collect **blocks, transactions, ERC‑20/721 tokens, transfers, receipts, logs, contracts and internal transactions**, and notes that the resulting data is available in BigQuery[\[2\]](https://github.com/blockchain-etl/ethereum-etl#:~:text=Python%20scripts%20for%20ETL%20,gl%2FoY5BCQ).  
* **How data is represented** – Each table is stored as a CSV/BigQuery table with schema information. Examples include:  
* **blocks** – includes block number, block hash, parent hash, miner address, difficulty, total difficulty, gas limits/used and timestamp[\[3\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=blocks).  
  * **transactions** – stores individual transaction records with fields such as transaction hash, block number, from and to address, value, gas, gas price, input data and additional fields for EIP‑1559 and blob gas[\[4\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=transactions).  
  * **token\_transfers** – a derived table containing ERC‑20/ERC‑721 transfer events; it records token address, sender and receiver addresses, transfer value, associated transaction hash and log index[\[5\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=token_transfers).  
  * **receipts** – contains transaction receipt data such as gas used, cumulative gas used, contract address created, transaction status and effective gas price[\[6\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=receipts).  
  * **logs** – captures all smart‑contract event logs; it includes log index, transaction hash, block number, the contract address that emitted the event, raw data and topics[\[7\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=logs).  
  * **contracts** – lists contract addresses with bytecode, function signature hashes and booleans indicating whether a contract implements ERC‑20 or ERC‑721 standards[\[8\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=contracts).  
  * **tokens** – provides metadata for tokens such as address, symbol, name, decimals, total supply and the block at which the token metadata was fetched[\[9\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=tokens).  
* **traces** – contains internal transactions and trace information including block number, transaction hash, from/to addresses, value, input/output data, trace type, call type, gas, gas used and error status[\[10\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=traces).  
* **Additional tables** – A tutorial describing how to query the dataset notes that there are **eleven** tables in total: tokens, blocks, contracts, traces, token\_transfers, balances, transactions, sessions, logs, load\_metadata and amended\_tokens[\[11\]](https://bitquery.io/blog/querying-bigquery-blockchain-dataset#:~:text=You%27ll%20be%20presented%20with%20a,dataset%2C%20which%20are%20as%20follows). The balances table holds Ether balances for every address and is updated daily. The amended\_tokens table deduplicates token metadata when CREATE2 causes duplicates. load\_metadata and sessions provide metadata about data loads and query sessions.  
* **Relationships & querying advice** – The dataset is highly relational.  
* **Transactions vs. traces** – The transactions table contains canonical Ethereum transactions, whereas *internal transactions* (those triggered inside smart‑contract calls) are only visible through the traces table. Internal transactions are not recorded directly on the blockchain and can only be seen via tracing.  
* **Contracts & transactions** – There is no unique key on the contracts table; when joining transactions and contracts, one must match on the block hash and either the to\_address or the receipt\_contract\_address.  
* **Token transfers vs. logs** – Token transfers table records only ERC‑20/721 transfer events, while the logs table stores **all** smart‑contract event logs. Logs are therefore used for general event reporting beyond token transfers.  
* **Balances** – The balances table keeps the Ether balance for each address and is refreshed daily. The Bitquery tutorial notes that the schema for this table has two fields, address (STRING) and eth\_balance (NUMERIC)[\[12\]](https://bitquery.io/blog/querying-bigquery-blockchain-dataset#:~:text=In%20the%20schema%2C%20we%20have,NUMERIC).  
* **Partitioning** – BigQuery tables such as transactions, logs, token\_transfers, traces and blocks are partitioned by timestamp fields; using partition filters (e.g., restricting by block\_timestamp or timestamp) improves query performance and reduces cost.  
* **Limitations & quirks** – The Ethereum ETL documentation highlights several caveats:  
* Proxy contracts can obscure interface detection; is\_erc20 and is\_erc721 flags will be **false** for proxy contracts, so some proxy token contracts will not appear in the tokens table[\[13\]](https://ethereum-etl.readthedocs.io/en/latest/limitations/#:~:text=,table).  
  * ERC‑20 metadata methods (symbol, name, decimals, total\_supply) are optional; about 10 % of contracts lack this data, and some contracts return values of the wrong type[\[14\]](https://ethereum-etl.readthedocs.io/en/latest/limitations/#:~:text=,in%20this%20case%20as%20well).  
  * Numeric values such as token\_transfers.value, tokens.decimals and tokens.total\_supply are stored as **STRING** because 32‑byte integers exceed BigQuery’s native numeric range; conversions to FLOAT64 can lose precision and conversions to NUMERIC may overflow[\[15\]](https://ethereum-etl.readthedocs.io/en/latest/limitations/#:~:text=,to%20convert%20to).  
  * Contracts without a decimals() method but with a fallback function returning a boolean will produce 0 or 1 in the decimals column[\[16\]](https://ethereum-etl.readthedocs.io/en/latest/limitations/#:~:text=,column%20in%20the%20CSVs).

## Key tables and keywords

| Table/field | Keyword(s) or short description |
| :---- | :---- |
| **blocks** | block number, hash, parent hash, gas used/limit, timestamp[\[3\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=blocks) |
| **transactions** | tx hash, from/to address, value, gas, gas price, EIP‑1559 fields[\[4\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=transactions) |
| **token\_transfers** | token address, sender, receiver, transfer value, log index[\[5\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=token_transfers) |
| **receipts** | cumulative gas used, gas used, contract address created, status[\[6\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=receipts) |
| **logs** | smart‑contract event log index, data, topics[\[7\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=logs) |
| **contracts** | contract bytecode, function signature hashes, ERC‑20 and ERC‑721 flags[\[8\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=contracts) |
| **tokens** | token metadata: address, symbol, name, decimals, total supply[\[9\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=tokens) |
| **traces** | internal transaction traces: from/to address, value, call type, gas used, error[\[10\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=traces) |
| **balances** | address, Ether balance (refreshed daily)[\[12\]](https://bitquery.io/blog/querying-bigquery-blockchain-dataset#:~:text=In%20the%20schema%2C%20we%20have,NUMERIC) |
| **amended\_tokens** | deduplicated token metadata for duplicate contract addresses |
| **sessions/load\_metadata** | metadata about BigQuery sessions and dataset loading[\[11\]](https://bitquery.io/blog/querying-bigquery-blockchain-dataset#:~:text=You%27ll%20be%20presented%20with%20a,dataset%2C%20which%20are%20as%20follows) |

## Usage notes

* **Comprehensive analytics** – With tables for native transactions, internal traces and smart‑contract events, the dataset enables a wide range of analyses such as tracing token flows, measuring network congestion, calculating gas usage and exploring DeFi or NFT contract interactions.

* **Joining tables** – Properly joining tables requires paying attention to keys (e.g., using composite keys when joining contracts with transactions) and understanding that internal transactions reside in the traces table.

* **Data quality considerations** – Users should account for the limitations described above (proxy contracts, missing metadata, string numeric fields). For numeric fields stored as strings, cast to numeric types carefully[\[15\]](https://ethereum-etl.readthedocs.io/en/latest/limitations/#:~:text=,to%20convert%20to).

* **Data access** – The dataset is free to query in Kaggle kernels and is mirrored from BigQuery. The update lag (\~4 minutes) means that block data may be slightly behind the latest chain state[\[1\]](https://ethereum-etl.readthedocs.io/en/latest/google-bigquery/#:~:text=Querying%20in%20BigQuery).

---

[\[1\]](https://ethereum-etl.readthedocs.io/en/latest/google-bigquery/#:~:text=Querying%20in%20BigQuery) Google BigQuery \- Ethereum ETL

[https://ethereum-etl.readthedocs.io/en/latest/google-bigquery/](https://ethereum-etl.readthedocs.io/en/latest/google-bigquery/)

[\[2\]](https://github.com/blockchain-etl/ethereum-etl#:~:text=Python%20scripts%20for%20ETL%20,gl%2FoY5BCQ) GitHub \- blockchain-etl/ethereum-etl: Python scripts for ETL (extract, transform and load) jobs for Ethereum blocks, transactions, ERC20 / ERC721 tokens, transfers, receipts, logs, contracts, internal transactions. Data is available in Google BigQuery https://goo.gl/oY5BCQ

[https://github.com/blockchain-etl/ethereum-etl](https://github.com/blockchain-etl/ethereum-etl)

[\[3\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=blocks) [\[4\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=transactions) [\[5\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=token_transfers) [\[6\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=receipts) [\[7\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=logs) [\[8\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=contracts) [\[9\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=tokens) [\[10\]](https://ethereum-etl.readthedocs.io/en/latest/schema/#:~:text=traces) Schema \- Ethereum ETL

[https://ethereum-etl.readthedocs.io/en/latest/schema/](https://ethereum-etl.readthedocs.io/en/latest/schema/)

[\[11\]](https://bitquery.io/blog/querying-bigquery-blockchain-dataset#:~:text=You%27ll%20be%20presented%20with%20a,dataset%2C%20which%20are%20as%20follows) [\[12\]](https://bitquery.io/blog/querying-bigquery-blockchain-dataset#:~:text=In%20the%20schema%2C%20we%20have,NUMERIC) Google BigQuery Ethereum Dataset: A Comprehensive Tutorial \- Bitquery

[https://bitquery.io/blog/querying-bigquery-blockchain-dataset](https://bitquery.io/blog/querying-bigquery-blockchain-dataset)

[\[13\]](https://ethereum-etl.readthedocs.io/en/latest/limitations/#:~:text=,table) [\[14\]](https://ethereum-etl.readthedocs.io/en/latest/limitations/#:~:text=,in%20this%20case%20as%20well) [\[15\]](https://ethereum-etl.readthedocs.io/en/latest/limitations/#:~:text=,to%20convert%20to) [\[16\]](https://ethereum-etl.readthedocs.io/en/latest/limitations/#:~:text=,column%20in%20the%20CSVs) Limitations \- Ethereum ETL

[https://ethereum-etl.readthedocs.io/en/latest/limitations/](https://ethereum-etl.readthedocs.io/en/latest/limitations/)
