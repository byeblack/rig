use std::collections::HashMap;

use futures::{stream, StreamExt, TryStreamExt};

use crate::{
    completion::{CompletionError, CompletionModel, Document, ToolDefinition},
    message::Message,
    vector_store::VectorStoreError,
};

use super::Agent;

pub trait ComputingDynamicInfo<M: CompletionModel> {
    fn computing_context(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> impl std::future::Future<Output = Result<Vec<Document>, CompletionError>> + Send;
    fn computing_tools(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> impl std::future::Future<Output = Result<Vec<ToolDefinition>, CompletionError>> + Send;
}

impl<M: CompletionModel> ComputingDynamicInfo<M> for Agent<M> {
    async fn computing_context(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> Result<Vec<Document>, CompletionError> {
        let prompt: Message = prompt.into();
        let Some(text) = prompt.rag_text() else {
            return Err(CompletionError::RequestError("Invalid prompt".into()));
        };

        let dynamic_context = stream::iter(self.dynamic_context.iter())
            .then(|(num_sample, index)| async {
                Ok::<_, VectorStoreError>(
                    index
                        .top_n(&text, *num_sample)
                        .await?
                        .into_iter()
                        .map(|(_, id, doc)| {
                            // Pretty print the document if possible for better readability
                            let text = serde_json::to_string_pretty(&doc)
                                .unwrap_or_else(|_| doc.to_string());

                            Document {
                                id,
                                text,
                                additional_props: HashMap::new(),
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .try_fold(vec![], |mut acc, docs| async {
                acc.extend(docs);
                Ok(acc)
            })
            .await
            .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

        Ok(dynamic_context)
    }

    async fn computing_tools(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> Result<Vec<ToolDefinition>, CompletionError> {
        let prompt: Message = prompt.into();
        let Some(text) = &prompt.rag_text() else {
            return Err(CompletionError::RequestError("Invalid prompt".into()));
        };

        let static_tools = stream::iter(self.static_tools.iter())
            .filter_map(|toolname| async move {
                if let Some(tool) = self.tools.get(toolname) {
                    Some(tool.definition(text.into()).await)
                } else {
                    tracing::warn!("Tool implementation not found in toolset: {}", toolname);
                    None
                }
            })
            .collect::<Vec<_>>()
            .await;

        let dynamic_tools = stream::iter(self.dynamic_tools.iter())
            .then(|(num_sample, index)| async {
                Ok::<_, VectorStoreError>(
                    index
                        .top_n_ids(text, *num_sample)
                        .await?
                        .into_iter()
                        .map(|(_, id)| id)
                        .collect::<Vec<_>>(),
                )
            })
            .try_fold(vec![], |mut acc, docs| async {
                for doc in docs {
                    if let Some(tool) = self.tools.get(&doc) {
                        acc.push(tool.definition(text.into()).await)
                    } else {
                        tracing::warn!("Tool implementation not found in toolset: {}", doc);
                    }
                }
                Ok(acc)
            })
            .await
            .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

        Ok([static_tools, dynamic_tools].concat())
    }
}
