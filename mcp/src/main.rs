use anyhow::Result;
use rmcp::{Content, ListToolsResponse, RequestId, ServerCapabilities, ServerInfo, Tool};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::info;

struct CounterServer {
    counter: Arc<AtomicU64>,
}

impl CounterServer {
    fn new() -> Self {
        Self {
            counter: Arc::new(AtomicU64::new(0)),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    info!("Starting minimal MCP server");
    
    let server = Arc::new(CounterServer::new());
    
    let server_info = ServerInfo {
        name: "ai-os-mcp".to_string(),
        version: "0.1.0".to_string(),
    };
    
    let capabilities = ServerCapabilities {
        tools: Some(json!({})),
        ..Default::default()
    };
    
    let tools = vec![
        Tool {
            name: "increment".to_string(),
            description: Some("Increment the counter by a specified amount".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                        "description": "Amount to increment"
                    }
                },
                "required": []
            }),
        },
        Tool {
            name: "decrement".to_string(),
            description: Some("Decrement the counter by a specified amount".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                        "description": "Amount to decrement"
                    }
                },
                "required": []
            }),
        },
        Tool {
            name: "get_value".to_string(),
            description: Some("Get the current counter value".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        Tool {
            name: "reset".to_string(),
            description: Some("Reset the counter to zero".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
    ];
    
    let mut mcp_server = rmcp::Server::new(server_info, capabilities);
    
    let server_clone = server.clone();
    mcp_server.on_list_tools(move |_req_id| {
        let tools = tools.clone();
        async move {
            Ok(ListToolsResponse { tools })
        }
    });
    
    let server_clone = server.clone();
    mcp_server.on_call_tool(move |request, _req_id: RequestId| {
        let server = server_clone.clone();
        async move {
            let tool_name = &request.name;
            let arguments = request.arguments.as_ref();
            
            match tool_name.as_str() {
                "increment" => {
                    let amount = arguments
                        .and_then(|a| a.get("amount"))
                        .and_then(|a| a.as_u64())
                        .unwrap_or(1);
                    
                    let new_value = server.counter.fetch_add(amount, Ordering::SeqCst) + amount;
                    
                    Ok(vec![Content::Text {
                        text: json!({
                            "action": "increment",
                            "amount": amount,
                            "new_value": new_value
                        }).to_string(),
                    }])
                }
                "decrement" => {
                    let amount = arguments
                        .and_then(|a| a.get("amount"))
                        .and_then(|a| a.as_u64())
                        .unwrap_or(1);
                    
                    let current = server.counter.load(Ordering::SeqCst);
                    let new_value = if current >= amount {
                        server.counter.fetch_sub(amount, Ordering::SeqCst) - amount
                    } else {
                        server.counter.store(0, Ordering::SeqCst);
                        0
                    };
                    
                    Ok(vec![Content::Text {
                        text: json!({
                            "action": "decrement",
                            "amount": amount,
                            "new_value": new_value
                        }).to_string(),
                    }])
                }
                "get_value" => {
                    let value = server.counter.load(Ordering::SeqCst);
                    
                    Ok(vec![Content::Text {
                        text: json!({
                            "value": value
                        }).to_string(),
                    }])
                }
                "reset" => {
                    server.counter.store(0, Ordering::SeqCst);
                    
                    Ok(vec![Content::Text {
                        text: json!({
                            "action": "reset",
                            "new_value": 0
                        }).to_string(),
                    }])
                }
                _ => Ok(vec![Content::Text {
                    text: format!("Unknown tool: {}", tool_name),
                }]),
            }
        }
    });
    
    info!("Minimal MCP server ready, listening on stdio");
    mcp_server.run_stdio().await?;
    
    Ok(())
}