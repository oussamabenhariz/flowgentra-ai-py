//! Python bindings for the skills system.
//!
//! Rust handles: SKILL.md parsing, menu building, system prompt generation,
//! tool-name validation.
//!
//! Python handles: scripts/ discovery (importlib/inspect) and storing Python
//! callables for skill-specific tools.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use flowgentra_ai::core::skills::{Skill, SkillRegistry};
use crate::error::to_py_err;

// ── PySkill ────────────────────────────────────────────────────────────────────

/// A fully parsed skill (read-only view exposed to Python).
#[pyclass(name = "Skill")]
#[derive(Clone)]
pub struct PySkill {
    inner: Skill,
}

#[pymethods]
impl PySkill {
    #[getter] fn name(&self) -> &str { &self.inner.name }
    #[getter] fn description(&self) -> &str { &self.inner.description }
    #[getter] fn version(&self) -> Option<&str> { self.inner.version.as_deref() }
    #[getter] fn license(&self) -> Option<&str> { self.inner.license.as_deref() }
    #[getter] fn instructions(&self) -> &str { &self.inner.instructions }
    #[getter] fn allowed_tools(&self) -> Vec<String> { self.inner.allowed_tools.clone() }
    #[getter] fn references(&self) -> Vec<String> { self.inner.references.clone() }

    fn __repr__(&self) -> String {
        format!("Skill(name='{}', allowed_tools={:?})", self.inner.name, self.inner.allowed_tools)
    }
}

// ── PySkillRegistry ────────────────────────────────────────────────────────────

/// Registry that manages skills and generates system prompts.
///
/// Follows the two-phase interaction model from the SKILLS_PROPOSAL:
///
/// **Phase 1 — Discovery**: call ``build_menu()`` — LLM sees only names + descriptions.
/// **Phase 2 — Execution**: call ``build_system_prompt(skill_name)`` + ``resolve_tools(skill_name)``.
///
/// Tool scoping: ``resolve_tools(skill_name)`` returns **only** the tools declared in
/// that skill's ``allowed-tools`` — all other tools are hidden from the LLM.
#[pyclass(name = "SkillRegistry")]
pub struct PySkillRegistry {
    inner: SkillRegistry,
    /// Skill-specific Python callables from scripts/.
    /// Key: tool name → callable decorated with @tool.
    skill_tools: HashMap<String, PyObject>,
    /// Python ToolRegistry wrapper (flowgentra_ai.tools.ToolRegistry).
    /// Used for built-in tool validation + ToolSpec resolution.
    tool_registry: Option<PyObject>,
}

#[pymethods]
impl PySkillRegistry {

    /// Create an empty registry.
    ///
    /// Args:
    ///     tool_registry: ``ToolRegistry`` instance (with built-ins + any shared
    ///                    custom tools). Used to validate ``allowed-tools`` names
    ///                    and resolve ToolSpec objects.
    #[new]
    #[pyo3(signature = (tool_registry=None))]
    fn new(tool_registry: Option<PyObject>) -> Self {
        PySkillRegistry {
            inner: SkillRegistry::new(),
            skill_tools: HashMap::new(),
            tool_registry,
        }
    }

    /// Scan a directory and load every subdirectory that contains a SKILL.md.
    ///
    /// Skill-specific tools in ``scripts/`` are auto-discovered for each skill.
    ///
    /// Args:
    ///     path:           Path to the skills directory.
    ///     tool_registry:  ``ToolRegistry`` with built-ins and shared custom tools.
    ///     allow_override: If True, replace duplicate skill names silently.
    #[staticmethod]
    #[pyo3(signature = (path, tool_registry=None, allow_override=false))]
    fn from_directory(
        py: Python<'_>,
        path: &str,
        tool_registry: Option<PyObject>,
        allow_override: bool,
    ) -> PyResult<Self> {
        let mut registry = PySkillRegistry::new(tool_registry);
        let skills_dir = PathBuf::from(path);

        if !skills_dir.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                format!("Skills directory not found: {path}"),
            ));
        }

        let mut entries: Vec<PathBuf> = std::fs::read_dir(&skills_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?
            .filter_map(|e| e.ok().map(|d| d.path()))
            .filter(|p| p.is_dir() && p.join("SKILL.md").exists())
            .collect();
        entries.sort();

        if entries.is_empty() {
            return Err(crate::error::ConfigurationError::new_err(format!(
                "No skills found in '{path}'. \
                 from_directory() scans for subdirectories that contain a SKILL.md file. \
                 To load a single skill directly, use: \
                 registry = SkillRegistry(tool_registry=...); registry.load('{path}')"
            )));
        }

        for entry in entries {
            registry.load(py, entry.to_str().unwrap_or(""), allow_override)?;
        }

        Ok(registry)
    }

    /// Load a single skill from a directory.
    ///
    /// 1. Parses ``SKILL.md`` (Rust).
    /// 2. Discovers ``scripts/`` tools via Python importlib.
    /// 3. Validates all ``allowed-tools`` names against known tools.
    /// 4. Registers the skill in the Rust registry.
    #[pyo3(signature = (path, allow_override=false))]
    fn load(&mut self, py: Python<'_>, path: &str, allow_override: bool) -> PyResult<()> {
        let skill_dir = PathBuf::from(path);
        let skill_md_path = skill_dir.join("SKILL.md");

        if !skill_md_path.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                format!("No SKILL.md found in {path}"),
            ));
        }

        let content = std::fs::read_to_string(&skill_md_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let references = load_references(&skill_dir);

        let dir_name = skill_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let skill = SkillRegistry::build_skill(&content, dir_name, references)
            .map_err(to_py_err)?;

        // Discover @tool decorated callables from scripts/
        let scripts_dir = skill_dir.join("scripts");
        let discovered = discover_script_tools(py, &scripts_dir)?;

        // Build the full set of known tool names for validation:
        //   skill-specific tools (from scripts/) + global ToolRegistry tools
        let mut known_tools: Vec<String> = discovered.keys().cloned().collect();
        if let Some(ref tr) = self.tool_registry {
            let names: Vec<String> = tr
                .call_method0(py, "list_names")?
                .extract::<Vec<String>>(py)?;
            known_tools.extend(names);
        }

        SkillRegistry::validate_tools(&skill, &known_tools).map_err(to_py_err)?;

        // Store skill-specific callables
        self.skill_tools.extend(discovered);

        self.inner.register(skill, allow_override).map_err(to_py_err)
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    fn list(&self) -> Vec<String> {
        self.inner.list().iter().map(|s| s.to_string()).collect()
    }

    fn get(&self, name: &str) -> PyResult<PySkill> {
        self.inner
            .get(name)
            .map(|s| PySkill { inner: s.clone() })
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "Skill '{name}' not found. Loaded: {:?}",
                    self.inner.list()
                ))
            })
    }

    fn __contains__(&self, name: &str) -> bool { self.inner.contains(name) }
    fn __len__(&self) -> usize { self.inner.len() }

    fn __repr__(&self) -> String {
        format!("SkillRegistry(skills={:?})", self.inner.list())
    }

    // ── Phase 1 — Discovery ───────────────────────────────────────────────────

    /// Return the discovery-phase system prompt.
    ///
    /// The LLM sees **only** skill names and descriptions — no instructions,
    /// no tools. It selects a skill via the ``activate_skill`` tool.
    fn build_menu(&self) -> String {
        self.inner.build_menu()
    }

    // ── Phase 2 — Execution ───────────────────────────────────────────────────

    /// Return the execution-phase system prompt for a skill (or all if None).
    ///
    /// Injects the skill's full instruction body and any reference content.
    #[pyo3(signature = (skill_name=None))]
    fn build_system_prompt(&self, skill_name: Option<&str>) -> PyResult<String> {
        self.inner.build_system_prompt(skill_name).map_err(to_py_err)
    }

    /// Return ``ToolSpec`` objects for a skill's ``allowed-tools``.
    ///
    /// Only the tools declared in the named skill's ``allowed-tools`` are
    /// returned — all other tools are hidden from the LLM.
    ///
    /// Resolves from two sources:
    ///   1. Skill-specific tools discovered from ``scripts/`` (``@tool`` decorated)
    ///   2. Global ``ToolRegistry`` tools (built-ins or shared custom tools)
    ///
    /// Args:
    ///     skill_name: Skill to resolve. Required — scoping is per-skill.
    fn resolve_tools(&self, py: Python<'_>, skill_name: &str) -> PyResult<PyObject> {
        let tool_names = self.inner.allowed_tools(skill_name).map_err(to_py_err)?;

        let agent_mod = py.import_bound("flowgentra_ai.agent")?;
        let tool_spec_cls = agent_mod.getattr("ToolSpec")?;
        let specs = PyList::empty_bound(py);

        for tool_name in tool_names {
            if let Some(callable) = self.skill_tools.get(tool_name) {
                // Skill-specific @tool callable — build ToolSpec from its metadata
                let name: String = callable.getattr(py, "_tool_name")?.extract(py)?;
                let desc: String = callable.getattr(py, "_tool_description")?.extract(py)?;
                let params: HashMap<String, String> = callable
                    .getattr(py, "_tool_parameters")?
                    .extract(py)?;
                let required: Vec<String> = callable
                    .getattr(py, "_tool_required")?
                    .extract(py)?;

                let spec = tool_spec_cls.call1((&name, &desc))?;
                for (param_name, param_type) in &params {
                    spec.call_method1("add_parameter", (param_name, param_type))?;
                }
                for req in &required {
                    spec.call_method1("set_required", (req,))?;
                }
                specs.append(spec)?;

            } else if let Some(ref tr) = self.tool_registry {
                // Global ToolRegistry tool (built-in or shared custom)
                let has: bool = tr
                    .call_method1(py, "has", (tool_name,))?
                    .extract(py)?;
                if has {
                    let defn: PyObject = tr.call_method1(py, "get", (tool_name,))?;
                    let name: String = defn
                        .call_method1(py, "__getitem__", ("name",))?
                        .extract(py)?;
                    let desc: String = defn
                        .call_method1(py, "__getitem__", ("description",))?
                        .extract(py)?;
                    let spec = tool_spec_cls.call1((&name, &desc))?;
                    specs.append(spec)?;
                }
            }
        }

        Ok(specs.into())
    }

    // ── Tool execution ────────────────────────────────────────────────────────

    /// Return the Python callable for a skill-specific tool, or None for built-ins.
    fn get_callable(&self, py: Python<'_>, tool_name: &str) -> Option<PyObject> {
        self.skill_tools.get(tool_name).map(|v| v.clone_ref(py))
    }

    /// Execute a skill-specific tool by name.
    ///
    /// For global ``ToolRegistry`` tools, use ``ToolRegistry.call_tool()`` directly.
    fn call_skill_tool(
        &self,
        py: Python<'_>,
        tool_name: &str,
        input: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let callable = self.skill_tools.get(tool_name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "No skill-specific tool '{tool_name}'. \
                 For global ToolRegistry tools use ToolRegistry.call_tool() directly."
            ))
        })?;
        callable.call_bound(py, (), Some(input))
    }
}

// ── Filesystem helpers ────────────────────────────────────────────────────────

fn load_references(skill_dir: &Path) -> Vec<String> {
    let references_dir = skill_dir.join("references");
    if !references_dir.exists() {
        return vec![];
    }
    let mut entries: Vec<PathBuf> = match std::fs::read_dir(&references_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|d| d.path()))
            .filter(|p| p.is_file())
            .collect(),
        Err(_) => return vec![],
    };
    entries.sort();
    entries
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .collect()
}

// ── Python-only: scripts/ discovery ───────────────────────────────────────────

/// Scan a skill's ``scripts/`` directory for ``@tool`` decorated functions.
///
/// Looks for functions with ``_is_tool = True`` — set by ``@tool`` from
/// ``flowgentra_ai.tools``. This must stay in Python-land because it uses
/// importlib to execute .py files and inspect Python function attributes.
fn discover_script_tools(py: Python<'_>, scripts_dir: &Path) -> PyResult<HashMap<String, PyObject>> {
    let mut tools: HashMap<String, PyObject> = HashMap::new();
    if !scripts_dir.exists() {
        return Ok(tools);
    }

    let mut py_files: Vec<PathBuf> = std::fs::read_dir(scripts_dir)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| p.extension().map_or(false, |ext| ext == "py"))
        .collect();
    py_files.sort();

    let importlib_util = py.import_bound("importlib.util")?;
    let inspect = py.import_bound("inspect")?;

    for py_file in py_files {
        let file_str = py_file.to_str().unwrap_or("");
        let stem = py_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("_skill_script");
        let module_name = format!("_flowgentra_skill_{stem}");

        let spec = importlib_util
            .call_method1("spec_from_file_location", (&module_name, file_str))?;
        if spec.is_none() {
            continue;
        }

        let module = importlib_util.call_method1("module_from_spec", (&spec,))?;
        spec.getattr("loader")?.call_method1("exec_module", (&module,))?;

        let members = inspect.call_method1(
            "getmembers",
            (&module, inspect.getattr("isfunction")?),
        )?;

        for item in members.iter()? {
            let item = item?;
            let func: PyObject = item.get_item(1)?.into();

            // Check for _is_tool attribute set by @tool decorator
            let is_tool = func
                .getattr(py, "_is_tool")
                .unwrap_or_else(|_| false.to_object(py));

            if is_tool.extract::<bool>(py).unwrap_or(false) {
                let name: String = func.getattr(py, "_tool_name")?.extract(py)?;
                tools.insert(name, func);
            }
        }
    }

    Ok(tools)
}
