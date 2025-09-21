import express from "express";
import dotenv from "dotenv";
import path from "path";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { DynamicStructuredTool } from "langchain/tools";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";

dotenv.config();
const port = 3000;
const app = express();
app.use(express.json());

const __dirname = path.resolve();

const model = new ChatGoogleGenerativeAI({
  model: "models/gemini-2.5-flash",
  maxOutputTokens: 2048,
  temperature: 0.7,
  apiKey: process.env.GOOGLE_API_KEY,    
});

const getMenuTool = new DynamicStructuredTool({
  name: "getMenuTool",
  description:
    "Returns the final answer for today's menu for the given category (breakfast, lunch, or dinner).",
  schema: z.object({
    category: z.string().describe("type of meal. Example: breakfast, lunch, dinner"),
  }),
  func: async ({ category }) => {
    const menus = {
      breakfast: "Allo Paratha, Poha, Masala Chai",
      lunch: "Paneer Butter Masala, Dal Fry, Jeera Rice, Roti",
      dinner: "Veg Biryani, Raita, Salad, Gulab Jamun",
    };
    return menus[category.toLowerCase()] || "No menu was found for that category.";
  },
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that uses tools when needed."],
  ["human", "{input}"],
  ["ai", "{agent_scratchpad}"],
]);
 
(async () => {
  const agent = await createToolCallingAgent({
    llm: model,
    tools: [getMenuTool],
    prompt,
  });

  const executor = await AgentExecutor.fromAgentAndTools({
    agent,
    tools: [getMenuTool],
    verbose: true,
    maxIterations: 3,
    returnIntermediateSteps: true,
  });

  app.post("/api/chat", async (req, res) => {
  const userInput = req.body.input;
  console.log("userInput: ", userInput);

  try { 
    if (["breakfast", "lunch", "dinner"].includes(userInput.toLowerCase())) {
      const data = await getMenuTool.func({ category: userInput });
      return res.json({ output: data });
    }
 
    const response = await executor.invoke({ input: userInput });
    console.log("Agent full response: ", response);

    if (response.output && response.output !== "Agent stopped due to max iterations.") {
      return res.json({ output: response.output });
    }

    res.status(500).json({ output: "Agent couldn't find a valid answer." });
  } catch (err) {
    console.error("Error during agent execution: ", err);
    res.status(500).json({ output: "Sorry, something went wrong. Please try again." });
  }
});


  app.get("/", (req, res) => {
    return res.sendFile(path.join(__dirname, "public", "index.html"));
  });

  app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
  });
})();
