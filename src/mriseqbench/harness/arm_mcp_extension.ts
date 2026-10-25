import { Type } from "typebox";
import { execFile } from "node:child_process";

// Pi extension for Julia MCP arms: Julia runs only through julia_eval; file
// tools remain. No Bash, generic subprocess tool or Julia launcher is exposed;
// benchmark controls are called from julia_eval with run(...).
const TOOLS = ["read", "write", "edit", "julia_eval", "julia_restart", "julia_list_sessions"];
const SIMULATOR = process.env.MRISEQBENCH_AGENT_SIMULATOR === "koma"
  ? " This combined arm provides the external KomaMRI simulator; call run(`./bin/simulate sequence.seq`) through julia_eval."
  : " No simulator is available.";

function bridge(name: string, args: unknown, signal: AbortSignal) {
  return new Promise<any>((resolve, reject) => {
    execFile("/opt/mcp-venv/bin/python", ["/arm_mcp_bridge.py", JSON.stringify({name, arguments: args})],
      {signal, maxBuffer: 2*1024*1024}, (error, stdout, stderr) => {
        if (error) reject(new Error(`${error.message}\n${stderr.slice(-2000)}`));
        else { try { resolve(JSON.parse(stdout)); } catch (error) { reject(error); } }
      });
  });
}

export default function(pi: any) {
  for (const [name, parameters, description] of [
    ["julia_eval", Type.Object({code: Type.String(), timeout: Type.Optional(Type.Number({minimum: 0, maximum: 1200}))}),
      "Run Julia code using aplavin/julia-mcp in the persistent KomaMRIBase author project. ALWAYS use this tool for Julia execution. The default timeout 0 relies on the harness charged-time deadline, so native grading can finish. Use println/display to see output; include(\"design.jl\") reads your file. State persists." + SIMULATOR],
    ["julia_restart", Type.Object({}), "Reset the MCP Julia session only as a last resort. Restart/setup time counts toward the same remaining trial budget."],
    ["julia_list_sessions", Type.Object({}), "List this trial's author-only MCP session."],
  ] as any[]) {
    pi.registerTool({name, label: name, description, parameters,
      async execute(_id: string, args: unknown, signal: AbortSignal) {
        const response = await bridge(name, args, signal);
        return {content: response.content, details: {transport: "MCP stdio", isError: response.isError ?? false}};
      }});
  }
  pi.on("session_start", async () => pi.setActiveTools(TOOLS));
  pi.on("tool_call", async (event: any) => {
    if (!TOOLS.includes(event.toolName))
      return {block: true, reason: "This arm executes code only through the pinned Julia MCP."};
  });
}
